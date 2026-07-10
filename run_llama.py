# Same runner, different default model
# python run_llama.py --in data/arxiv_csml_body_abstract.csv --out output/Abstracts_LLaMA3_8B_Instruct.csv

import argparse
import time
from run_qwen import main as run_main

if __name__ == "__main__":
    # allow overriding via CLI, but default to llama
    import sys
    
    ## This is the exact model name
    sys.argv += ["--model", "meta-llama-3.1-8b-instruct"]
    #sys.argv += ["--model", "meta-llama-3.1-8b-instruct-gguf"]
    
    # Measure total processing time
    start_time = time.perf_counter()
    
    run_main()
    
    end_time = time.perf_counter()
    elapsed_sec = end_time - start_time
    print(f"\nTotal processing time: {elapsed_sec:.2f} seconds")
