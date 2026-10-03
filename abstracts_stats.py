#!/usr/bin/env python3
"""
abstracts_stats.py

Purpose:
- Calculate per-abstract statistics for generated abstracts so the resulting CSV
  can be used for statistical significance testing. Each article has its own row.

Input format:
- --gen CSV: columns [paper_id, gen_abstract]
- --gold CSV: columns [paper_id, title, abstract, body]

Output format:
- CSV saved to stats/ with name derived from --gen: e.g.
  Gen_Abstracts_Mixtral_Instruct_7B_S1.csv -> stats/Gen_Abstracts_Mixtral_Instruct_7B_S1_Stats.csv
- DataFrame stats_df with columns in order: paper_id, word_len, BLEU, ROUGE-1, ROUGE-L, BERTScore

Preprocessing:
- Reuse existing light normalization from evaluate_abstracts._normalize_text:
  1. Convert None to empty string.
  2. Replace \r\n and \r with \n.
  3. Strip leading/trailing whitespace.
  4. Collapse multiple whitespace characters into a single space.
- Apply same normalization to generated and gold abstracts.
- Additionally remove known LLM introductory boilerplate from generated abstract:
  "Here is a combined abstract that focuses on motivation, methods, and main results:"
  (exact string as common; handled in dedicated clean function).
- Do not remove legitimate content.

Metric implementations (preserve project conventions):
- BLEU: per-article. Implementation choice documented below; reuses sacrebleu (as in evaluate_abstracts)
- ROUGE-1 F1, ROUGE-L F1: per-article using rouge_score; use_stemmer=True (same as evaluate_abstracts)
- BERTScore F1: per-article using bert_score; model_type=roberta-large, lang=en,
  rescale_with_baseline=False, values converted to percentage (same as evaluate_abstracts)

Notes:
- Do not aggregate across articles; one value per row.
- Keep paper_id consistent for cross-matching S1/S2/S3.
- No p-values/hypothesis testing.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import pandas as pd


def _normalize_text(s: str) -> str:
    """
    Reuse existing light normalization from evaluate_abstracts.py.
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    s = " ".join(s.split())
    return s


_BOILERPLATE_PREFIX = "Here is a combined abstract that focuses on motivation, methods, and main results:"


def _clean_generated_abstract(s: str) -> str:
    """
    Remove introductory/preamble boilerplate from generated abstracts.

    Order within this function:
    1. If None/empty, return normalized/empty form consistent with normalization.
    2. Apply existing normalization to ensure consistent whitespace.
    3. Remove introductory boilerplate using colon-based heuristic.
    4. Normalize whitespace again and return.

    Colon-based rule:
    - If the text contains a colon, remove everything from the beginning
      through the first colon, including the colon itself, but only when
      the text before the first colon appears to be an introductory/preamble
      statement. The heuristic below preserves the complete abstract when
      the first colon is clearly part of the actual abstract content.

    Heuristic conditions to treat as boilerplate (all must hold):
    - There is at least one colon.
    - The prefix before the first colon is relatively short (<= 120 chars).
    - The prefix starts with common introductory phrases (case-insensitive):
      'here is', 'this is', 'the following', 'below is', 'we present',
      'we propose', 'i will', 'let me', 'here are'
    - OR more generally, the prefix is clearly a preamble: it contains words
      like 'combined abstract', 'abstract that', 'summary', 'generated',
      'output', 'response', 'answer', or ends with typical preamble markers
      ('focuses on', 'integrates', 'provides').
    - The body after the colon (stripped) is non-empty.
    """
    if s is None:
        return ""
    text = str(s)

    # Step 1 & 2: normalize
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = " ".join(text.split())

    if not text:
        return text

    # Step 3: colon-based boilerplate removal with heuristic
    if ":" in text:
        colon_idx = text.find(":")
        prefix = text[:colon_idx]
        body = text[colon_idx + 1 :].strip()

        if body:  # don't split if nothing follows
            prefix_lower = prefix.lower()
            short_prefix = len(prefix) <= 120

            is_intro = False
            if short_prefix:
                intro_patterns = [
                    "here is",
                    "here are",
                    "this is",
                    "the following",
                    "below is",
                    "we present",
                    "we propose",
                    "i will",
                    "let me",
                ]
                if any(p in prefix_lower for p in intro_patterns):
                    is_intro = True
                preamble_markers = [
                    "combined abstract",
                    "abstract that",
                    "summary",
                    "generated",
                    "output",
                    "response",
                    "answer",
                    "focuses on",
                    "integrates",
                    "provides",
                ]
                if any(m in prefix_lower for m in preamble_markers):
                    is_intro = True

            if is_intro:
                # Treat prefix+colon as boilerplate
                text = body

    # Step 4: normalize again
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = " ".join(text.split())
    return text


def _read_csv(path: str, required_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Found: {list(df.columns)}")
    return df


def _merge_on_paper_id(gen_df: pd.DataFrame, gold_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Match on paper_id. Convert to string, deduplicate keeping last (consistent with evaluate_abstracts),
    left-merge gold abstract; count missing gold.
    """
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


def _word_len(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def compute_bleu_per_article(cands: List[str], refs: List[str]) -> List[float]:
    """
    Compute BLEU per article (not corpus-level). Reuses sacrebleu's sentence_bleu
    for single-reference per sample, matching evaluate_abstracts' use of sacrebleu
    family. Note: corpus_bleu aggregates across all references/candidates; for
    per-article we compute sentence-level BLEU against the corresponding reference.
    """
    from sacrebleu import sentence_bleu

    scores = []
    for c, r in zip(cands, refs):
        # sentence_bleu expects reference list of strings
        try:
            b = sentence_bleu(c, [r])
            scores.append(float(b.score))
        except Exception:
            scores.append(0.0)
    return scores


def compute_rouge_per_article(cands: List[str], refs: List[str], use_stemmer: bool = True) -> Tuple[List[float], List[float]]:
    """
    Per-article ROUGE-1 F1 and ROUGE-L F1. Uses rouge_score with use_stemmer=True
    (consistent with evaluate_abstracts). Returns values as percentages.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=use_stemmer)
    r1_list = []
    rl_list = []
    for c, r in zip(cands, refs):
        try:
            s = scorer.score(r, c)  # (reference, candidate)
            r1_list.append(float(s["rouge1"].fmeasure) * 100.0)
            rl_list.append(float(s["rougeL"].fmeasure) * 100.0)
        except Exception:
            r1_list.append(0.0)
            rl_list.append(0.0)
    return r1_list, rl_list


def compute_bertscore_per_article(
    cands: List[str],
    refs: List[str],
    model_type: str,
    lang: str,
    device: str | None,
    batch_size: int,
) -> List[float]:
    """
    Per-article BERTScore F1. Uses bert_score with settings matching evaluate_abstracts:
    model_type=roberta-large (default), lang=en, rescale_with_baseline=False, verbose=False.
    Values converted to percentage.
    """
    from bert_score import score as bert_score

    if len(cands) == 0:
        return []

    p, r, f1 = bert_score(
        cands,
        refs,
        model_type=model_type,
        lang=lang,
        device=device,
        batch_size=batch_size,
        verbose=False,
        rescale_with_baseline=False,
    )
    # f1 is tensor; convert each element to percentage
    f1_list = [float(x.item()) * 100.0 for x in f1]
    return f1_list


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute per-article stats for generated abstracts.")
    p.add_argument("--gen", required=True, help="Path to generated CSV with columns: paper_id, gen_abstract")
    p.add_argument("--gold", required=True, help="Path to gold CSV with columns: paper_id, title, abstract, body")
    p.add_argument("--gen-text-col", default="gen_abstract", help="Generated text column name in --gen CSV")
    p.add_argument("--gold-text-col", default="abstract", help="Gold text column name in --gold CSV")

    p.add_argument("--bertscore-model", default="roberta-large", help="BERTScore model_type")
    p.add_argument("--bertscore-lang", default="en", help="BERTScore language hint")
    p.add_argument("--bertscore-device", default=None, help='Device like "cuda" or "cpu"; default lets bert-score decide')
    p.add_argument("--bertscore-batch-size", type=int, default=16, help="BERTScore batch size")
    p.add_argument("--no-rouge-stemmer", action="store_true", help="Disable stemming in ROUGE scoring")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    # Input validation and reading
    gen_df = _read_csv(args.gen, ["paper_id", args.gen_text_col])
    gold_df = _read_csv(args.gold, ["paper_id", args.gold_text_col])

    gen_df = gen_df.rename(columns={args.gen_text_col: "gen_abstract"})
    gold_df = gold_df.rename(columns={args.gold_text_col: "abstract"})

    # Normalization
    gen_df["gen_abstract"] = gen_df["gen_abstract"].map(_normalize_text)
    gold_df["abstract"] = gold_df["abstract"].map(_normalize_text)

    # Clean boilerplate from generated abstracts (normalize + colon-based removal)
    gen_df["gen_abstract_clean"] = gen_df["gen_abstract"].map(_clean_generated_abstract)

    # Merge
    merged, n_missing_gold = _merge_on_paper_id(
        gen_df[["paper_id", "gen_abstract", "gen_abstract_clean"]],
        gold_df[["paper_id", "abstract"]],
    )

    cands_clean = merged["gen_abstract_clean"].tolist()
    refs = merged["abstract"].tolist()
    paper_ids = merged["paper_id"].tolist()

    n_empty_gen = int((merged["gen_abstract_clean"] == "").sum())
    n_empty_gold = int((merged["abstract"] == "").sum())

    # Compute per-article metrics
    bleu_list = compute_bleu_per_article(cands_clean, refs)
    rouge1_list, rougel_list = compute_rouge_per_article(cands_clean, refs, use_stemmer=not args.no_rouge_stemmer)
    bert_list = compute_bertscore_per_article(
        cands=cands_clean,
        refs=refs,
        model_type=args.bertscore_model,
        lang=args.bertscore_lang,
        device=args.bertscore_device,
        batch_size=args.bertscore_batch_size,
    )

    # word_len per article from cleaned generated
    word_len_list = [_word_len(t) for t in cands_clean]

    # Build stats_df with exact columns and order
    stats_df = pd.DataFrame(
        {
            "paper_id": paper_ids,
            "word_len": word_len_list,
            "BLEU": bleu_list,
            "ROUGE-1": rouge1_list,
            "ROUGE-L": rougel_list,
            "BERTScore": bert_list,
        }
    )

    # Output validation
    required_cols_out = ["paper_id", "word_len", "BLEU", "ROUGE-1", "ROUGE-L", "BERTScore"]
    missing_out = [c for c in required_cols_out if c not in stats_df.columns]
    if missing_out:
        raise ValueError(f"Missing required output columns: {missing_out}")
    if len(stats_df) != len(merged):
        raise ValueError(f"Row count mismatch: stats_df has {len(stats_df)} rows, merged has {len(merged)}")

    # Save
    stats_dir = os.path.join(os.getcwd(), "stats")
    os.makedirs(stats_dir, exist_ok=True)
    gen_basename = os.path.basename(args.gen)
    out_basename = gen_basename.replace(".csv", "_Stats.csv")
    out_path = os.path.join(stats_dir, out_basename)

    stats_df.to_csv(out_path, index=False)

    # Summary
    print("Generated rows:", len(gen_df))
    print("Gold rows:", len(gold_df))
    print("Matched rows:", len(merged))
    print("Missing gold:", n_missing_gold)
    print("Empty generated:", n_empty_gen)
    print("Empty gold:", n_empty_gold)
    print("\nStatistics:")
    print(stats_df.head().to_string(index=False))
    print("\nSaved:")
    print(out_path)


if __name__ == "__main__":
    main()
