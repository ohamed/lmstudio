# prep_arxiv_csml_pairs.py
# Usage:
#   python prep_arxiv_csml_pairs.py /path/to/s2orc_csml_jsonl_or_jsonl.gz out.csv

import sys, os, gzip, json, csv

IN_PATH = sys.argv[1]
OUT_CSV = sys.argv[2]

def open_any(path):
    return gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') else open(path, 'r', encoding='utf-8')

def is_csml(rec):
    # Accept if clearly CS/ML by arXiv category or field labels (fallback to True if unknown)
    cat = (rec.get('arxiv_id') or '')
    fields = [f.lower() for f in rec.get('mag_field_of_study') or []]
    if cat.startswith('cs.') or cat.startswith('stat.ml'):
        return True
    if any('computer' in f or 'machine' in f or 'artificial intelligence' in f for f in fields):
        return True
    return False if (cat or fields) else True

def join_body(rec):
    body = rec.get('body_text') or []
    if isinstance(body, list):
        texts = []
        for x in body:
            t = x.get('text') if isinstance(x, dict) else str(x)
            if t: texts.append(t.strip())
        return '\n\n'.join(texts).strip()
    # some variants store as string
    if isinstance(body, str):
        return body.strip()
    return ''

def clean(s):
    return ' '.join((s or '').split())

def stream_records(path):
    with open_any(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, list):
                # yield each element if list contains dicts
                for item in rec:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(rec, dict):
                yield rec

def main():
    sample_count = 0
    with open_any(IN_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, list):
                for item in rec:
                    if isinstance(item, dict):
                        print(json.dumps(item, indent=2)[:500])  # preview first record
                        sample_count += 1
                        if sample_count >= 3:
                            return
            elif isinstance(rec, dict):
                print(json.dumps(rec, indent=2)[:500])
                sample_count += 1
                if sample_count >= 3:
                    return

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python prep_arxiv_csml_pairs.py data/S2ORC_title_abstract.jsonl.gz data/output.csv")
        sys.exit(1)
    main()
