# ingest_infohub_bge_m3_full.py
# pip install requests beautifulsoup4 chromadb markdownify FlagEmbedding tqdm

import os, re, time, hashlib, pickle
import requests
from markdownify import markdownify as md
import chromadb
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

# ── Config ────────────────────────────────────────────────────────────────────

LANG        = "ka"
BASE_API    = "https://infohubapi.rs.ge"

EMBED_MODEL = "BAAI/bge-m3"
DB_DIR      = "chroma_infohub_bgem3_full"
COLLECTION  = "infohub_ka"
CORPUS_PATH = "bgem3_corpus_full.pkl"

TYPEID_MIN          = 1
TYPEID_MAX          = 80
TAKE                = 98
SLEEP_SEC           = 0.12
MAX_PAGES_PER_TYPE  = 80
CHUNK_MAX_CHARS     = 1400
CHUNK_OVERLAP_SENTS = 2
IGNORE_ITEM_IDS     = "22337"

# ── Session ───────────────────────────────────────────────────────────────────

def make_session():
    sess = requests.Session()
    sess.headers.update({
        "User-Agent":      "Mozilla/5.0",
        "Accept":          "application/json, text/plain, */*",
        "Referer":         "https://infohub.rs.ge/",
        "Origin":          "https://infohub.rs.ge",
        "Accept-Language": "en-US,en;q=0.9",
        "Languagecode":    LANG,
    })
    return sess

# ── API helpers ───────────────────────────────────────────────────────────────

def safe_get_json(sess, url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

def also_search(sess, type_id, skip, take=TAKE):
    return safe_get_json(sess, f"{BASE_API}/api/AlsoSearch", params={
        "skip": str(skip), "take": str(take),
        "typeId": str(type_id), "ignoreItemIds": IGNORE_ITEM_IDS,
    })

def get_arr_and_total(data):
    arr, total = [], None
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            arr = data["data"]
        for k in ["total", "totalCount", "count", "recordsTotal", "itemsCount", "totalItems"]:
            if isinstance(data.get(k), int):
                total = data[k]
                break
        if total is None and isinstance(data.get("meta"), dict):
            for k in ["total", "totalCount", "count", "itemsCount", "totalItems"]:
                if isinstance(data["meta"].get(k), int):
                    total = data["meta"][k]
                    break
    return arr, total

def fetch_details(sess, key: str):
    return safe_get_json(sess, f"{BASE_API}/api/documents/{key}/details-by-key",
                         params={"openFromSearch": "false"})

# ── Type ID + key discovery ───────────────────────────────────────────────────

def discover_all_types(sess) -> list[dict]:
    """
    Your proven approach: probe each typeId via AlsoSearch,
    page through all results, collect every uniqueKey.
    Returns list of {"id", "name", "total", "keys"}.
    """
    print(f"Probing type IDs {TYPEID_MIN}–{TYPEID_MAX} …\n")
    valid = []

    for type_id in tqdm(range(TYPEID_MIN, TYPEID_MAX + 1), desc="Scanning type IDs"):
        try:
            data = also_search(sess, type_id, 0)
            arr, _ = get_arr_and_total(data)
            if not arr:
                continue

            # Get type name from first result
            name = f"Type_{type_id}"
            if isinstance(arr[0].get("type"), dict):
                name = arr[0]["type"].get("name") or name

            # Page through all results and collect keys
            keys, seen, skip = [], set(), 0
            for page in range(MAX_PAGES_PER_TYPE):
                page_data       = also_search(sess, type_id, skip)
                page_arr, _     = get_arr_and_total(page_data)

                if not page_arr:
                    break

                added = 0
                for it in page_arr:
                    if not isinstance(it, dict):
                        continue
                    k = str(it.get("uniqueKey", ""))
                    if k and k not in seen:
                        seen.add(k)
                        keys.append(k)
                        added += 1

                if added == 0 or len(page_arr) < TAKE:
                    break
                skip += TAKE
                time.sleep(SLEEP_SEC)

            if keys:
                valid.append({"id": type_id, "name": name, "total": len(keys), "keys": keys})
                tqdm.write(f"  typeId={type_id:3d}  docs={len(keys):5d}  {name}")

        except Exception as e:
            tqdm.write(f"  typeId={type_id:3d}  error: {str(e)[:100]}")

        time.sleep(SLEEP_SEC)

    return valid

# ── Text processing ───────────────────────────────────────────────────────────

def html_to_markdown(x: str) -> str:
    if not x or not isinstance(x, str):
        return ""
    text = md(x, heading_style="ATX", bullets="-")
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def sentence_chunks(text: str, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP_SENTS):
    text = (text or "").strip()
    if not text:
        return []
    sents = re.split(r"(?<=[\.\!\?\:\;])\s+|\n{2,}", text)
    sents = [s.strip() for s in sents if s.strip()]
    out, cur, cur_len = [], [], 0
    for s in sents:
        if cur_len + len(s) + 1 > max_chars and cur:
            out.append(" ".join(cur).strip())
            cur     = cur[-overlap:]
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        out.append(" ".join(cur).strip())
    return out

def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def clean_meta(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out

def extract_article_number(title: str, text: str) -> str:
    pattern = r"(?:მუხლი|Article|§|Clause)\s*\d+[\.\d]*"
    for src in (title, text[:200]):
        m = re.search(pattern, src, re.IGNORECASE)
        if m:
            return m.group(0)
    return ""

def extract_text_and_meta(data: dict):
    title = (data.get("name") or data.get("nameLink") or "").strip()
    parts = []
    for raw in (data.get("description") or "", data.get("additionalDescription") or ""):
        if isinstance(raw, str) and raw.strip():
            parts.append(html_to_markdown(raw) if ("<" in raw and ">" in raw) else raw.strip())
    text = "\n\n".join(p for p in parts if p.strip())

    tp            = data.get("type") or {}
    doc_type_id   = tp.get("id")   if isinstance(tp, dict) else None
    doc_type_name = tp.get("name") if isinstance(tp, dict) else None

    meta = clean_meta({
        "uniqueKey":   data.get("uniqueKey"),
        "published":   bool(data.get("published")) if data.get("published") is not None else None,
        "docTypeId":   doc_type_id,
        "docTypeName": doc_type_name,
        "baseType":    str(data.get("baseType")) if data.get("baseType") is not None else None,
        "views":       int(data.get("views")) if isinstance(data.get("views"), int) else None,
        "title":       title,
        "chapter":     str(data.get("chapterName") or data.get("chapter") or ""),
        "section":     str(data.get("sectionName") or data.get("section") or ""),
    })
    return title, text, meta

# ── BGE-M3 embedding ──────────────────────────────────────────────────────────

class BGEM3EmbedFunction:
    def __init__(self):
        self.model = BGEM3FlagModel(EMBED_MODEL, use_fp16=True)

    def __call__(self, input):
        out = self.model.encode(input, batch_size=32, max_length=512,
                                return_dense=True, return_sparse=False,
                                return_colbert_vecs=False)
        return out["dense_vecs"].tolist()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sess     = make_session()
    embed_fn = BGEM3EmbedFunction()

    client = chromadb.PersistentClient(path=DB_DIR)
    col    = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"source": "infohub.rs.ge", "lang": LANG,
                  "embed_model": EMBED_MODEL, "hnsw:space": "cosine"},
    )

    # Resume support
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "rb") as f:
            corpus = pickle.load(f)
        print(f"Resuming — {len(corpus)} chunks already embedded\n")
    else:
        corpus = {}

    print("db_dir:     ", os.path.abspath(DB_DIR))
    print("embed_model:", EMBED_MODEL)
    print()

    # ── Step 1: Discover all type IDs and keys ────────────────────────────────
    all_types = discover_all_types(sess)

    if not all_types:
        print("No valid type IDs found. Check your network/API.")
        return

    grand_total = sum(t["total"] for t in all_types)
    print(f"\n── {len(all_types)} types found | {grand_total} total documents ──\n")
    for t in all_types:
        print(f"  typeId={t['id']:3d}  docs={t['total']:5d}  {t['name']}")
    print()

    # ── Step 2: Embed all documents ───────────────────────────────────────────
    global_ok = global_fail = 0

    for t in all_types:
        tid   = t["id"]
        tname = t["name"]
        keys  = t["keys"]

        print(f"\n── typeId={tid} | {tname} | {len(keys)} docs ──────────────────")

        ok = fail = 0
        doc_pbar = tqdm(keys, desc="  Embedding", unit="doc", leave=True)

        for key in doc_pbar:
            try:
                data              = fetch_details(sess, key)
                title, text, meta = extract_text_and_meta(data)

                if not text.strip():
                    fail += 1
                    time.sleep(SLEEP_SEC)
                    continue

                chunks = sentence_chunks(text)
                if not chunks:
                    fail += 1
                    time.sleep(SLEEP_SEC)
                    continue

                doc_type    = meta.get("docTypeName") or tname
                article_num = extract_article_number(title, text)
                chapter     = meta.get("chapter") or ""
                section     = meta.get("section") or ""

                ids, docs, metas = [], [], []
                for j, chunk in enumerate(chunks):
                    cid = f"{key}:{j}:{sha(chunk)[:12]}"

                    if cid in corpus:
                        continue  # already done, skip

                    ctx_lines = [f"Title: {title}", f"Type: {doc_type}"]
                    if article_num: ctx_lines.append(f"Article: {article_num}")
                    if chapter:     ctx_lines.append(f"Chapter: {chapter}")
                    if section:     ctx_lines.append(f"Section: {section}")
                    contextualized = "\n".join(ctx_lines) + "\n\n" + chunk

                    ids.append(cid)
                    docs.append(contextualized)
                    corpus[cid] = contextualized

                    chunk_meta = dict(meta)
                    chunk_meta.update({
                        "chunk":               j,
                        "chunk_total":         len(chunks),
                        "article_num":         article_num,
                        "original_chunk_text": chunk,
                    })
                    metas.append(clean_meta(chunk_meta))

                if ids:
                    col.upsert(ids=ids, documents=docs, metadatas=metas)

                ok += 1
                doc_pbar.set_postfix(ok=ok, fail=fail, db_chunks=col.count())

            except Exception as e:
                fail += 1
                tqdm.write(f"  FAIL key={key}  err={str(e)[:120]}")

            time.sleep(SLEEP_SEC)

        doc_pbar.close()
        global_ok   += ok
        global_fail += fail
        tqdm.write(f"  done typeId={tid}  ok={ok}  fail={fail}  db_chunks={col.count()}")

        # Checkpoint after every type — safe to interrupt and resume
        with open(CORPUS_PATH, "wb") as f:
            pickle.dump(corpus, f)

    print("\n── Ingestion complete ───────────────────────────────────────────")
    print(f"ok_docs:            {global_ok}")
    print(f"fail_docs:          {global_fail}")
    print(f"total_chunks_in_db: {col.count()}")
    print(f"db_dir:             {os.path.abspath(DB_DIR)}")
    print(f"corpus:             {os.path.abspath(CORPUS_PATH)}")

if __name__ == "__main__":
    main()