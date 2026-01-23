#!/usr/bin/env python3
"""
evaluate_abstracts.py

Compute corpus-level metrics for generated abstracts against gold abstracts.

Metrics
- BLEU (SacreBLEU)
- ROUGE-1 F1 (rouge_score)
- ROUGE-L F1 (rouge_score)
- BERTScore F1 (bert_score)

Install 
- the dependencies in the requirements first, then:

pip install sacrebleu rouge-score bert-score

- BERTScore uses PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

Run
python evaluate_abstracts.py \
  --gen output/Gen_Abstracts_Qwen3_8B_Instruct_b50.csv \
  --gold data/arxiv_csml_body_abstract.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class EvalSummary:
    n_gen_rows: int
    n_gold_rows: int
    n_merged: int
    n_missing_gold: int
    n_empty_gen: int
    n_empty_gold: int
    bleu: float
    rouge1_f1: float
    rougel_f1: float
    bertscore_f1: float


def _read_csv(path: str, required_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Found: {list(df.columns)}")
    return df


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    s = " ".join(s.split())
    return s


def _merge_on_paper_id(gen_df: pd.DataFrame, gold_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    gen_df = gen_df.copy()
    gold_df = gold_df.copy()

    gen_df["paper_id"] = gen_df["paper_id"].astype(str)
    gold_df["paper_id"] = gold_df["paper_id"].astype(str)

    gen_df = gen_df.drop_duplicates(subset=["paper_id"], keep="last")
    gold_df = gold_df.drop_duplicates(subset=["paper_id"], keep="last")

    merged = gen_df.merge(gold_df[["paper_id", "abstract"]], on="paper_id", how="left")
    n_missing_gold = int(merged["abstract"].isna().sum())
    merged["abstract"] = merged["abstract"].fillna("")
    return merged, n_missing_gold


def compute_bleu(cands: List[str], refs: List[str]) -> float:
    from sacrebleu import corpus_bleu

    bleu = corpus_bleu(cands, [refs])
    return float(bleu.score)


def compute_rouge_f1(cands: List[str], refs: List[str], use_stemmer: bool = True) -> Tuple[float, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=use_stemmer)

    r1_sum = 0.0
    rl_sum = 0.0
    n = 0

    for c, r in zip(cands, refs):
        scores = scorer.score(r, c)
        r1_sum += float(scores["rouge1"].fmeasure)
        rl_sum += float(scores["rougeL"].fmeasure)
        n += 1

    if n == 0:
        return 0.0, 0.0

    return r1_sum / n, rl_sum / n


def compute_bertscore_f1(
    cands: List[str],
    refs: List[str],
    model_type: str,
    lang: str,
    device: str | None,
    batch_size: int,
) -> float:
    from bert_score import score as bert_score

    p, r, f1 = bert_score(
        cands,
        refs,
        model_type=model_type,
        lang=lang,
        device=device,
        batch_size=batch_size,
        verbose=False,
        rescale_with_baseline=False, #True
    )
    return float(f1.mean().item())


def evaluate(
    gen_path: str,
    gold_path: str,
    gen_text_col: str,
    gold_text_col: str,
    bert_model: str,
    bert_lang: str,
    bert_device: str | None,
    bert_batch_size: int,
    rouge_stemmer: bool,
) -> EvalSummary:
    gen_df = _read_csv(gen_path, ["paper_id", gen_text_col])
    gold_df = _read_csv(gold_path, ["paper_id", gold_text_col])

    gen_df = gen_df.rename(columns={gen_text_col: "gen_abstract"})
    gold_df = gold_df.rename(columns={gold_text_col: "abstract"})

    gen_df["gen_abstract"] = gen_df["gen_abstract"].map(_normalize_text)
    gold_df["abstract"] = gold_df["abstract"].map(_normalize_text)

    merged, n_missing_gold = _merge_on_paper_id(gen_df[["paper_id", "gen_abstract"]], gold_df[["paper_id", "abstract"]])

    cands = merged["gen_abstract"].tolist()
    refs = merged["abstract"].tolist()

    n_empty_gen = int((merged["gen_abstract"] == "").sum())
    n_empty_gold = int((merged["abstract"] == "").sum())

    bleu = compute_bleu(cands, refs)
    rouge1_f1, rougel_f1 = compute_rouge_f1(cands, refs, use_stemmer=rouge_stemmer)
    bert_f1 = compute_bertscore_f1(
        cands=cands,
        refs=refs,
        model_type=bert_model,
        lang=bert_lang,
        device=bert_device,
        batch_size=bert_batch_size,
    )

    return EvalSummary(
        n_gen_rows=int(len(gen_df)),
        n_gold_rows=int(len(gold_df)),
        n_merged=int(len(merged)),
        n_missing_gold=n_missing_gold,
        n_empty_gen=n_empty_gen,
        n_empty_gold=n_empty_gold,
        bleu=bleu,
        rouge1_f1=rouge1_f1,
        rougel_f1=rougel_f1,
        bertscore_f1=bert_f1,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate generated abstracts against gold abstracts.")
    p.add_argument("--gen", required=True, help="Path to generated CSV with columns: paper_id, gen_abstract")
    p.add_argument("--gold", required=True, help="Path to gold CSV with columns: paper_id, abstract")
    p.add_argument("--gen-text-col", default="gen_abstract", help="Generated text column name in --gen CSV")
    p.add_argument("--gold-text-col", default="abstract", help="Gold text column name in --gold CSV")

    p.add_argument("--bertscore-model", default="roberta-large", help="BERTScore model_type")
    p.add_argument("--bertscore-lang", default="en", help="BERTScore language hint")
    p.add_argument("--bertscore-device", default=None, help='Device, like "cuda" or "cpu". Default lets bert-score decide')
    p.add_argument("--bertscore-batch-size", type=int, default=16, help="BERTScore batch size")

    p.add_argument("--no-rouge-stemmer", action="store_true", help="Disable stemming in ROUGE scoring")

    p.add_argument("--json", dest="json_out", default=None, help="Optional path to save results as JSON")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    summary = evaluate(
        gen_path=args.gen,
        gold_path=args.gold,
        gen_text_col=args.gen_text_col,
        gold_text_col=args.gold_text_col,
        bert_model=args.bertscore_model,
        bert_lang=args.bertscore_lang,
        bert_device=args.bertscore_device,
        bert_batch_size=args.bertscore_batch_size,
        rouge_stemmer=not args.no_rouge_stemmer,
    )

    print("Files")
    print(f"gen  {args.gen}")
    print(f"gold {args.gold}")
    print("")
    print("Counts")
    print(f"generated rows            {summary.n_gen_rows}")
    print(f"gold rows                 {summary.n_gold_rows}")
    print(f"matched on paper_id       {summary.n_merged}")
    print(f"generated missing gold    {summary.n_missing_gold}")
    print(f"empty generated abstracts {summary.n_empty_gen}")
    print(f"empty gold abstracts      {summary.n_empty_gold}")
    print("")
    print("Scores")
    print(f"BLEU           {summary.bleu:.4f}")
    print(f"ROUGE-1 F1     {summary.rouge1_f1:.6f}")
    print(f"ROUGE-L F1     {summary.rougel_f1:.6f}")
    print(f"BERTScore F1   {summary.bertscore_f1:.6f}")

    if args.json_out:
        payload = {
            "gen_path": args.gen,
            "gold_path": args.gold,
            "counts": {
                "n_gen_rows": summary.n_gen_rows,
                "n_gold_rows": summary.n_gold_rows,
                "n_merged": summary.n_merged,
                "n_missing_gold": summary.n_missing_gold,
                "n_empty_gen": summary.n_empty_gen,
                "n_empty_gold": summary.n_empty_gold,
            },
            "scores": {
                "bleu": summary.bleu,
                "rouge1_f1": summary.rouge1_f1,
                "rougel_f1": summary.rougel_f1,
                "bertscore_f1": summary.bertscore_f1,
            },
            "settings": {
                "gen_text_col": args.gen_text_col,
                "gold_text_col": args.gold_text_col,
                "bertscore_model": args.bertscore_model,
                "bertscore_lang": args.bertscore_lang,
                "bertscore_device": args.bertscore_device,
                "bertscore_batch_size": args.bertscore_batch_size,
                "rouge_stemmer": not args.no_rouge_stemmer,
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("")
        print(f"Saved JSON {args.json_out}")


if __name__ == "__main__":
    main()
