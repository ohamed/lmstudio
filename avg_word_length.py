"""

python avg_word_length.py --csv data/arxiv_csml_body_abstract.csv --column abstract

python avg_word_length.py --csv output/gen/Gen_Abstracts_Qwen3_8B_Instruct_b50_S1.csv --column gen_abstract

python avg_word_length.py --csv output/gen/Gen_Abstracts_Qwen3_8B_Instruct_S2.csv --column gen_abstract

python avg_word_length.py --csv output/gen/Gen_Abstracts_LLaMA3_8B_Instruct_b50_S1.csv --column gen_abstract

python avg_word_length.py --csv output/gen/Gen_Abstracts_LLaMA3_8B_Instruct_S3.csv --column gen_abstract

python avg_word_length.py --csv output/gen/Gen_Abstracts_Mixtral_Instruct_7B_S1.csv --column gen_abstract

python avg_word_length.py --csv output/gen/Gen_Abstracts_LLaMA3_8B_Instruct.csv --column gen_abstract

"""

import argparse
import pandas as pd

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
    
    # Clean the text column: remove the specific prefix phrase
    #Here is a new formal academic abstract of exactly 120-130 words in one continuous paragraph
    df[args.column] = df[args.column].astype(str).str.replace(
        "Here is a combined abstract that focuses on motivation, methods, and main results:", 
        "", 
        regex=False
    )

    # Drop missing values in the chosen column
    abstracts = df[args.column].dropna().astype(str)

    # Compute word lengths
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
