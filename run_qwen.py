# Main loop for Qwen 3 8B Instruct
# Usage:
#   python run_qwen.py --in data/arxiv_csml_body_abstract.csv --out Abstracts_Qwen3_8B_Instruct.csv
import argparse
import os
from lm_client import create_client, gen_abstract
from data_io import read_input, write_output, make_run_logger
from summarizer import map_reduce

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_csv", default="Abstracts_Qwen3_8B_Instruct.csv")
    ap.add_argument("--base_url", default="http://localhost:1234/v1")
    ap.add_argument("--model", default="qwen3-8b-instruct")
    ap.add_argument("--max_input_tokens", type=int, default=2500)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    log_path, log_f = make_run_logger()
    print(f"Logging to {log_path}")

    client = create_client(args.base_url)

    df = read_input(args.in_csv)
    rows = []
    total = len(df) if args.limit == 0 else min(args.limit, len(df))

    for idx, row in df.iterrows():
        if args.limit and len(rows) >= args.limit:
            break
        pid = str(row["paper_id"])
        body = str(row["body"] or "")
        if not body.strip():
            log_f.write(f"{pid}\tSKIP\tempty body\n")
            continue
        try:
            gen_abs = map_reduce(client, args.model, body, gen_abstract, max_input_tokens=args.max_input_tokens)
            rows.append((pid, gen_abs))
            log_f.write(f"{pid}\tOK\tlen={len(gen_abs)}\n")
        except Exception as e:
            log_f.write(f"{pid}\tERR\t{repr(e)}\n")
        if len(rows) % 10 == 0:
            print(f"Processed {len(rows)}/{total}")

    write_output(rows, args.out_csv)
    log_f.close()
    print(f"Saved {len(rows)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()
