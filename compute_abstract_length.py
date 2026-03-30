import pandas as pd

# Load dataset
df = pd.read_csv("data/arxiv_csml_body_abstract.csv")

# Drop missing abstracts
abstracts = df["abstract"].dropna().astype(str)

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