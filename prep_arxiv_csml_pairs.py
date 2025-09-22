# prep_csml_pairs.py
# Usage:
#   python prep_csml_pairs.py data/input.jsonl.gz out.csv

# pip install pandas requests tqdm pymupdf
import json, re, os, requests, fitz, pandas as pd
from tqdm import tqdm

META = "data/arxiv-metadata-oai-snapshot.json"
OUT  = "data/arxiv_csml_body_abstract.csv"
MAX_DOCS = 200  # stop after this many papers (set higher for full run)

def is_csml(cats):
    s = (cats or "").lower()
    return ("cs." in s) or ("stat.ml" in s)

def pdf_to_text(fp):
    try:
        doc = fitz.open(fp)
        txt = "\n".join(page.get_text() for page in doc)
        doc.close()
        return re.sub(r"\s+", " ", txt).strip()
    except:
        return ""

os.makedirs("data/pdfs", exist_ok=True)
rows = []

with open(META, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Scanning metadata"):
        rec = json.loads(line)
        cats = rec.get("categories","")
        if not is_csml(cats):
            continue
        arxiv_id = rec.get("id","").strip()
        title = (rec.get("title") or "").strip()
        abstract = re.sub(r"\s+"," ", (rec.get("abstract") or "").strip())
        if len(abstract) < 50:
            continue
        
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_fp = f"data/pdfs/{arxiv_id.replace('/','_')}.pdf"
        if not os.path.exists(pdf_fp):
            r = requests.get(pdf_url, timeout=60)
            if r.status_code == 200:
                with open(pdf_fp,"wb") as w: w.write(r.content)
            else:
                continue
        
        body = pdf_to_text(pdf_fp)
        if len(body) < 500:
            continue

        rows.append((arxiv_id, title, abstract, body))
        if len(rows) >= MAX_DOCS:
            break

df = pd.DataFrame(rows, columns=["paper_id","title","abstract","body"])
df.to_csv(OUT, index=False, encoding="utf-8")
print(f"Saved {len(df)} rows -> {OUT}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python prep_arxiv_csml_pairs.py data/S2ORC_title_abstract.jsonl.gz data/output.csv")
        sys.exit(1)
    main()
