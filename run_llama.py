# Same runner, different default model
# python run_llama.py --in data/...csv --out Abstracts_LLaMA3_8B_Instruct.csv
import argparse
from run_qwen import main as run_main

if __name__ == "__main__":
    # allow overriding via CLI, but default to llama
    import sys
    sys.argv += ["--model", "meta-llama-3.1-8b-instruct"]
    run_main()
