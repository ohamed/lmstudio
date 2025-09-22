# python run_mixtral.py --in data/...csv --out Abstracts_Mixtral_8x7B_Instruct.csv
import argparse
from run_qwen import main as run_main

if __name__ == "__main__":
    import sys
    sys.argv += ["--model", "mixtral-8x7b-instruct"]
    run_main()
