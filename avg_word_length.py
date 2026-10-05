"""
## Usage examples:
python avg_word_length.py --csv data/arxiv_csml_body_abstract.csv --column abstract

python avg_word_length.py --csv output/gen/Gen_Abstracts_Qwen3_8B_Instruct_b50.csv --column gen_abstract

python avg_word_length.py --csv output/gen/Gen_Abstracts_Qwen3_8B_Instruct_S2.csv --column gen_abstract

python avg_word_length.py --csv output/gen/Gen_Abstracts_LLaMA3_8B_Instruct_b50.csv --column gen_abstract

python avg_word_length.py --csv output/gen/Gen_Abstracts_LLaMA3_8B_Instruct_S3.csv --column gen_abstract

python avg_word_length.py --csv output/gen/Gen_Abstracts_Mixtral_Instruct_7B_S1.csv --column gen_abstract

python avg_word_length.py --csv output/gen/Gen_Abstracts_LLaMA3_8B_Instruct.csv --column gen_abstract

"""

"""
Compute descriptive statistics for the word lengths of text entries in a CSV file.

This script provides a command-line interface for analyzing the length
distribution of textual data stored in a specified CSV column. It loads the
input dataset, applies the same text preprocessing as abstracts_stats.py
(normalization for all; additional boilerplate removal only when column is gen_abstract),
ignores missing entries, and computes the average, median, minimum, and
maximum number of words per text instance.
"""

import argparse
import pandas as pd


def _normalize_text(s: str) -> str:
    """
    Normalize text same as abstracts_stats._normalize_text.
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    s = " ".join(s.split())
    return s


def _clean_generated_abstract(s: str) -> str:
    """
    Remove introductory boilerplate for generated abstracts same as abstracts_stats.
    Only removes 'Here is a' / 'Here's a' prefix through first colon if present.
    """
    if s is None:
        return ""

    text = _normalize_text(s)
    if not text:
        return text

    lower_text = text.lower()
    if lower_text.startswith("here is a ") or lower_text.startswith("here's a "):
        colon_idx = text.find(":")
        if colon_idx != -1:
            text = text[colon_idx + 1 :]
            text = _normalize_text(text)

    return text


def _preprocess_text(s: str, column_name: str) -> str:
    """
    Apply same preprocessing as abstracts_stats:
    - Always normalize.
    - If column is gen_abstract, also apply boilerplate removal.
    """
    if column_name == "gen_abstract":
        return _clean_generated_abstract(s)
    return _normalize_text(s)


def main():
    parser = argparse.ArgumentParser(
        description="Compute basic word-length stats for a text column in a CSV file."
    )
    parser.add_argument(
        "--csv",
        "-f",
        default="data/arxiv_csml_body_abstract.csv",
        help="Path to the CSV file (default: %(default)s)",
    )
    parser.add_argument(
        "--column",
        "-c",
        default="abstract",
        help="Name of the text column to analyze (default: %(default)s)",
    )

    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.csv)  # [web:18]

    # Preprocess using same logic as abstracts_stats
    df[args.column] = df[args.column].apply(lambda x: _preprocess_text(x, args.column))

    # Drop missing values in the chosen column
    abstracts = df[args.column].dropna().astype(str)

    # Compute word lengths (unchanged calculation)
    lengths = abstracts.apply(lambda x: len(x.split()))
    avg_len = lengths.mean()
    median_len = lengths.median()
    min_len = lengths.min()
    max_len = lengths.max()

    print(f"Average length: {avg_len:.0f} words")
    print(f"Median length: {median_len:.0f} words")
    print(f"Min length: {min_len} words")
    print(f"Max length: {max_len} words")


if __name__ == "__main__":
    main()
