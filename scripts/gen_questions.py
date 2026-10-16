"""Everyday Turkish user questions from source chunks (data plan step 1). COSTS MONEY.

Run from the repo root: python scripts/gen_questions.py --limit N [--per-chunk 5] [--tag v1]

Source chunks: data/chunks_<chunks>.jsonl (default v2), frozen by scripts/split_chunks.py before any question exists: the
usable kvkk-rag corpus chunks first, then chunks of the generation-only KVKK guides round-robin over
documents (exclusion rules in scripts/datagen.py). One LLM call per chunk, in that order, --per-chunk (3-5)
questions per call. --limit N is the number of questions to generate, so the run makes ceil(N / per-chunk)
calls (pilot: --limit 10 -> 2 calls).

The generator is not the teacher, so it may differ from it: gemini-3.8-flash, temperature 1.0 for variety,
thinking level LOW, JSON output. Every question then goes through the leakage filter (word 3-gram Jaccard
and e5 cosine against all eval questions, scripts/datagen.py); leaked ones are kept in the file with
"leak": true for the record and are never rewritten or trained on.

Appends to data/questions_<tag>.jsonl; chunks already in the file are skipped, so a run can be resumed.
Before any call: checks GEMINI_API_KEY in .env, prints the number of calls and the estimated cost, and
waits for "yes".
"""
import argparse
import json
import sys

import datagen
import kvkk_rag
import llm

PURPOSE = "gen_questions"
GEN = {"temperature": 1.0, "thinking_level": "LOW", "json_output": True}
# per-call cost before this project has measured its own: ~850 input tokens (prompt + 192-word chunk at
# ~2.2 tokens per word, as in kvkk-rag's answer calls), ~600 output tokens with LOW thinking; max 1000 / 1500
DEFAULT_MEAN = llm.cost_usd(850, 600)
DEFAULT_MAX = llm.cost_usd(1000, 1500)

# v1: the pilot's 10 questions. v2 (after the pilot): the questions should look like what real people type
# into a KVKK assistant or a search box, including short, sloppy and unaccented ones, and not quiz the passage;
# the model may return fewer questions, or none, when the passage holds nothing a real person would ask.
PROMPT_VERSION = "v2"
SYSTEM = """Görevin, Türkiye'deki kişisel verilerin korunması mevzuatından alınmış bir metin parçasını okuyup, hukuk bilgisi olmayan gerçek insanların bu konuda bir yapay zekâ asistanına veya arama kutusuna yazacağı soruları üretmek.

Soran kişi kim olabilir: bir çalışan, iş arayan biri, bir müşteri, bir hasta, bir veli, bir öğrenci, apartman yöneticisi, küçük bir işletme sahibi, insan kaynakları çalışanı, muhasebeci, yazılımcı, e-ticaret sitesi sahibi.

Kurallar:
1. Her soru, metindeki bilgiyle cevaplanabilmeli.
2. Gerçek bir ihtiyaçtan doğmalı: metni hiç okumamış biri bu soruyu gerçekten sorar mıydı? Sormazsa yazma. Metni sınav gibi sorgulayan sorular yazma ("kaç tane ... vardır", "hangi birimin görevidir" gibi).
3. Gündelik dil kullan. Metindeki hukuki terimleri ("ilgili kişi", "veri sorumlusu", "açık rıza", "anonim hale getirme", "özel nitelikli kişisel veri", "aydınlatma yükümlülüğü" gibi) kullanma; insanlar bunların yerine "bilgilerim", "şirket", "izin", "silmek", "sağlık bilgisi" gibi kelimeler kullanır. Madde numarası veya kanun, yönetmelik, rehber adı yazma ("kvkk" demek olur).
4. Soruları gerçek kullanıcılar gibi karıştır:
   - Bazıları kendi durumunu 1-3 cümleyle anlatıp sorsun ("Çalıştığım yerde ... yaptılar, bu normal mi?").
   - Bazıları kısa ve doğrudan olsun ("... isteyebilirler mi", "... silmek zorunda mı").
   - En az biri özensiz yazılsın: küçük harf, noktalama yok, bir iki yazım hatası veya Türkçe karakter yerine ı->i, ş->s gibi.
5. Sorular birbirinin tekrarı olmasın ve metinden cümle kopyalamasın.
6. Metinde gerçek bir kullanıcının soracağı bir şey yoksa daha az soru yaz; hiç yoksa boş liste döndür.
7. Yalnızca JSON döndür: {"sorular": ["...", "..."]}"""


def user_prompt(chunk: dict, k: int) -> str:
    return f"Metin parçası:\n\"\"\"\n{chunk['text']}\n\"\"\"\n\nBu metne göre en fazla {k} soru yaz."


def parse_questions(text: str) -> list[str] | None:
    """The questions, [] when the model found nothing worth asking, None when the output is not valid JSON."""
    try:
        qs = json.loads(text)["sorular"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return [" ".join(q.split()) for q in qs if isinstance(q, str) and q.strip()]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, required=True, help="generate about N questions (pilot: 10)")
    parser.add_argument("--per-chunk", type=int, default=5, choices=(3, 4, 5))
    parser.add_argument("--tag", default="v1", help="questions file: data/questions_<tag>.jsonl")
    parser.add_argument("--chunks", default="v2", help="frozen chunk split: data/chunks_<chunks>.jsonl")
    parser.add_argument("--test-v3", action="store_true", help="gate 7: the leakage filter checks eval/test_v3.jsonl too")
    parser.add_argument("--skip-chunks-of", nargs="*", default=[], metavar="TAG",
                        help="gate 7: skip chunks that already have questions in data/questions_<TAG>.jsonl")
    args = parser.parse_args()
    llm.api_key()
    if args.test_v3:
        datagen.add_test_v3()

    out_path = kvkk_rag.ROOT / f"data/questions_{args.tag}.jsonl"
    chunks_path = kvkk_rag.ROOT / f"data/chunks_{args.chunks}.jsonl"
    if not chunks_path.exists():
        sys.exit(f"{chunks_path} not found; run scripts/split_chunks.py first")
    chunks = [c for c in map(json.loads, chunks_path.read_text(encoding="utf-8").splitlines()) if c]
    order = sorted((c for c in chunks if c["keep"]), key=lambda c: c["order"])
    rows = []
    if out_path.exists():
        rows = [r for r in map(json.loads, out_path.read_text(encoding="utf-8").splitlines()) if r]
    done_chunks = {r["chunk_id"] for r in rows}
    seen = {r["question"] for r in rows if r["question"]}
    for tag in args.skip_chunks_of:
        other = [r for r in map(json.loads, (kvkk_rag.ROOT / f"data/questions_{tag}.jsonl").read_text(
            encoding="utf-8").splitlines()) if r]
        done_chunks |= {r["chunk_id"] for r in other}
        seen |= {r["question"] for r in other if r["question"]}
    todo = [c for c in order if c["chunk_id"] not in done_chunks][:llm.n_calls(args.limit, args.per_chunk)]
    print(f"chunks: {len(chunks)} total, {len(order)} usable, {len(done_chunks)} already done")
    if not todo:
        sys.exit("nothing to do: every usable chunk already has questions")

    mean, mx = llm.per_call_estimate(PURPOSE, DEFAULT_MEAN, DEFAULT_MAX)
    approved = llm.confirm_spend(f"Question generation ({len(todo)} chunks x {args.per_chunk} questions)",
                                 len(todo), mean, mx)
    cap = llm.RunCap(approved)
    leak_filter = datagen.LeakageFilter()
    n_ok = n_leak = 0
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for i, c in enumerate(todo, 1):
            out = llm.generate(user_prompt(c, args.per_chunk), SYSTEM, purpose=PURPOSE, **GEN)
            parsed = parse_questions(out["text"])
            qs = [q for q in parsed or [] if q not in seen][:args.per_chunk]
            if not qs:
                # a marker row, so the chunk counts as done and is not sent again on resume
                reason = "invalid output" if parsed is None else "no realistic question"
                print(f"[{i}/{len(todo)}] chunk {c['chunk_id']}: {reason}: {out['text'][:200]!r}")
                f.write(json.dumps({"qid": None, "chunk_id": c["chunk_id"], "question": None, "skipped": reason,
                                    "prompt": PROMPT_VERSION}, ensure_ascii=False) + "\n")
            for j, (q, leak) in enumerate(zip(qs, leak_filter.check(qs) if qs else []), 1):
                row = {"qid": f"{c['chunk_id']}_{j}", "chunk_id": c["chunk_id"], "source": c["source"], "doc": c["doc"],
                       "section": c["section"], "sections": c["sections"], "question": q, **leak,
                       "model": llm.MODEL, "prompt": PROMPT_VERSION,
                       **{k: v for k, v in GEN.items() if k != "json_output"}}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen.add(q)
                n_leak += leak["leak"]
                n_ok += not leak["leak"]
            f.flush()
            print(f"[{i}/{len(todo)}] chunk {c['chunk_id']} ({c['doc']}, {c['section']}): {len(qs)} questions, "
                  f"${out['cost_usd']:.4f}")
            cap.add(out["cost_usd"])
    print(f"done: {n_ok} questions kept, {n_leak} dropped by the leakage filter; run cost ${cap.spent:.4f}; "
          f"project total ${llm.spent_usd():.4f} of ${llm.BUDGET_USD:.2f}")


if __name__ == "__main__":
    main()
