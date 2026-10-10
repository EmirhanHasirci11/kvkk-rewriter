"""Download the generation-only texts: official KVKK guides that are not in kvkk-rag's retrieval corpus. No LLM call.

Run from the repo root: python scripts/fetch_gen_corpus.py

Source list: every PDF on https://www.kvkk.gov.tr/Icerik/2030/rehberler (5 pages, read 2026-10-10), in the
site's order. Rule fixed before reading any of them: take all, except the guides kvkk-rag's retrieval corpus
already holds in any version (special-category guide, data security guide). Nothing is dropped by content
here; chunks are filtered later by the same eval exclusion as the corpus chunks (scripts/datagen.py).

These texts are only used to generate training questions. They are never added to the retrieval corpus.
PDFs go to gen_corpus/raw/ (git-ignored); gen_corpus/sources.json records URL, size and sha256 of each file.
"""
import hashlib
import http.client
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import kvkk_rag

BASE = "https://www.kvkk.gov.tr/SharedFolderServer/CMSFiles/"
# (position on the guides page, title, file); "Yeni - " prefixes dropped
GUIDES = [
    (1, "Avukatların Mesleki Faaliyetlerinde Kişisel Verilerin Korunmasına İlişkin Uygulama Rehberi", "MTZhYjI0ZTE1NWIwMWM.pdf"),
    (2, "Kamu Kurum ve Kuruluşları ile Kamu Kurumu Niteliğindeki Meslek Kuruluşları İçin Kişisel Verilerin Korunması Kanununa Uyum Rehberi", "MTZhNjc1YWFhNmVjOGY.pdf"),
    (3, "Canlı Yayımlanan Reality Show Niteliğindeki Programlarda Kişisel Verilerin İşlenmesine Yönelik Rehber", "MTZhMDMwNmYyNjU3ZWY.pdf"),
    (4, "Üretken Yapay Zekâ ve Kişisel Verilerin Korunması Rehberi", "MTY5MjNmNmIwZWY3YTE.pdf"),
    (5, "Ödeme ve Elektronik Para Sektöründe Kişisel Verilerin Korunmasına İlişkin İyi Uygulamalar Rehberi", "5a7a3f8a-3b20-4773-970a-01aecfcc5222.pdf"),
    (6, "Özel Nitelikli Kişisel Verilerin İşlenmesine İlişkin Rehber", "70f95c73-06a2-44dc-81e9-34201bdd7f5c.pdf"),
    (7, "Kişisel Verilerin Yurt Dışına Aktarılması Rehberi", "13711235-abb6-4b17-9a6b-0a68c1ad86c5.pdf"),
    (8, "Seçim Faaliyetlerinde Kişisel Verilerin Korunması Rehberi", "a44a446f-43e9-434d-9426-b066fffdc349.pdf"),
    (9, "Türkiye Cumhuriyeti Kimlik Numaralarının İşlenmesi Hakkında Rehber", "a828de56-b2b2-43bf-b77d-cc05de5b2c84.pdf"),
    (10, "Mobil Uygulamalarda Mahremiyetin Korunmasına Yönelik Tavsiyeler", "8ba209bb-fa93-4479-84f0-dd55aac97a0f.pdf"),
    (11, "Genetik Verilerin İşlenmesinde Dikkat Edilmesi Gereken Hususlara İlişkin Rehber", "703442e0-690c-4618-91c3-83e7583170ca.pdf"),
    (12, "Kişisel Verilerin Korunmasına İlişkin Bankacılık Sektörü İyi Uygulamalar Rehberi", "12236bad-8de1-4c94-aad6-bb93f53271fb.pdf"),
    (13, "Çerez Uygulamaları Hakkında Rehber", "fb193dbb-b159-4221-8a7b-3addc083d33f.pdf"),
    (14, "Unutulma Hakkı (Unutulma Hakkının Arama Motorları Özelinde Değerlendirilmesi)", "11b6fd99-d42a-45b1-a009-21f2d36ded21.pdf"),
    (15, "Biyometrik Verilerin İşlenmesinde Dikkat Edilmesi Gereken Hususlara İlişkin Rehber", "bd06f5f4-e8cc-487e-abe1-d32dc18e2d7e.pdf"),
    (16, "Yapay Zeka Alanında Kişisel Verilerin Korunmasına Dair Tavsiyeler", "25a1162f-0e61-4a43-98d0-3e7d057ac31a.pdf"),
    (17, "Kişisel Veri İşleme Envanteri Hazırlama Rehberi", "14a17049-71da-4417-a07b-961f75a0070e.PDF"),
    (18, "Aydınlatma Yükümlülüğünün Yerine Getirilmesi Rehberi", "a569a068-c079-4189-b134-f57bc727af7d.pdf"),
    (19, "100 Soruda Kişisel Verilerin Korunması Kanunu", "185c2130-8070-4b2b-a91e-1d48322ca352.pdf"),
    (20, "Kişisel Verilerin Korunması Kanununa İlişkin Uygulama Rehberi", "MTY5MjZiZWNiNjQ3N2I.pdf"),
    (21, "Kişisel Veri Güvenliği Rehberi (Teknik ve İdari Tedbirler)", "7512d0d4-f345-41cb-bc5b-8d5cf125e3a1.pdf"),
    (22, "Kişisel Verilerin Silinmesi, Yok Edilmesi veya Anonim Hale Getirilmesi Rehberi", "bc1cb353-ef85-4e58-bb99-3bba31258508.pdf"),
    (23, "Kişisel Verilerin Korunması Kanunu Hakkında Sıkça Sorulan Sorular", "c148901d-cd7f-43ff-ae9a-978fb78fb43c.pdf"),
    (24, "Anayasal Bir Hak Olarak Kişisel Verilerin Korunmasını İsteme Hakkı", "28b8639c-1624-40af-ac9b-8f2551b4b2bf.pdf"),
    (25, "Kanun Kapsamındaki Hak ve Yükümlülükler", "8662692a-80ff-49a8-9ce9-7aca5b69ccd7.pdf"),
    (26, "Kişisel Verilerin İşlenme Şartları", "9feefe58-9b0f-49c9-a0c7-2b2eb8c012bb.pdf"),
    (27, "Kişisel Verilerin İşlenmesine İlişkin Temel İlkeler", "32ff74f6-9798-405a-b3d2-b42d28423fde.pdf"),
    (28, "Açık Rıza", "e3c6aa10-9de4-46f8-9b51-71bcf07c09b5.pdf"),
    (29, "6698 Sayılı Kanun'da Yer Alan Temel Kavramlar", "b414e1cf-bd67-43f1-a6f9-3bff081f92e7.pdf"),
    (30, "6698 Sayılı Kanun'da Yer Alan Terimler", "a9eb73e7-bf35-46d7-84e7-58a4618128cc.pdf"),
    (31, "6698 Sayılı Kişisel Verilerin Korunması Kanununun Amacı ve Kapsamı", "4d1c6565-b087-4bfb-b148-70e75b2549a5.pdf"),
    (32, "Kişisel Verilerin Korunması Alanında Uluslararası ve Ulusal Düzenlemeler", "49712b6d-655c-47e2-8c53-0c51bbef0306.pdf"),
    (33, "Kişisel Verilerin Korunması Kanununa Duyulan İhtiyaç", "6875e980-312d-4ed8-b2dc-9430891f2d8a.pdf"),
    (34, "Veri Sorumluları Sicili", "ed040488-ee55-40de-b503-646c8e1bc916.pdf"),
    (35, "İlgili Kişinin Hak Arama Yöntemleri", "353d85d5-87df-4c91-b08e-b4f0991a80f4.pdf"),
    (36, "Veri Sorumlusu ve Veri İşleyen", "31d9c444-27a5-4a75-95b1-1ca9bdb81ea5.pdf"),
    (37, "Kişisel Verileri Koruma Kurulu'nun Yapısı ve Görevleri", "c6206722-f4dc-484d-9bff-63f99b840158.pdf"),
]
IN_RETRIEVAL_CORPUS = {6: "rehber_ozel_nitelikli", 21: "rehber_veri_guvenligi"}

RAW = kvkk_rag.ROOT / "gen_corpus/raw"
SOURCES = kvkk_rag.ROOT / "gen_corpus/sources.json"


def doc_id(pos: int) -> str:
    return f"g{pos:02d}"


def download(url: str, path, tries: int = 8):
    """To <path>.part, resuming with an HTTP Range request after a dropped connection (the site is slow and
    some guides are ~100 MB); renamed to <path> only when complete."""
    part = path.with_suffix(".part")
    for attempt in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0", **({"Range": f"bytes={have}-"} if have else {})}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                if have and r.status != 206:     # server ignored the range: start over
                    have = 0
                    part.unlink()
                total = have + int(r.headers.get("Content-Length", 0))
                with part.open("ab") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            if part.stat().st_size >= total:
                part.rename(path)
                return
        except (OSError, http.client.HTTPException) as e:
            print(f"  {path.name}: attempt {attempt} stopped at {part.stat().st_size if part.exists() else 0} bytes ({e!r})")
    raise RuntimeError(f"{path.name}: download failed after {tries} attempts")


def fetch(item):
    pos, title, name = item
    path = RAW / f"{doc_id(pos)}.pdf"
    if not path.exists():
        download(BASE + name, path)
    data = path.read_bytes()
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"{path.name}: not a PDF ({data[:40]!r})")
    return {"id": doc_id(pos), "position": pos, "title": title, "url": BASE + name,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [g for g in GUIDES if g[0] not in IN_RETRIEVAL_CORPUS]
    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(fetch, todo))
    SOURCES.write_text(json.dumps({"index": "https://www.kvkk.gov.tr/Icerik/2030/rehberler", "read": "2026-10-10",
                                   "skipped_in_retrieval_corpus": IN_RETRIEVAL_CORPUS, "documents": rows},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(r["id"], f"{r['bytes'] / 1e6:5.1f} MB", r["title"][:80])
    print(f"{len(rows)} PDFs in {RAW}")


if __name__ == "__main__":
    main()
