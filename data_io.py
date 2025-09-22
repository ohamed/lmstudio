# IO utilities
import os
import pandas as pd
from datetime import datetime

def read_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"paper_id", "title", "abstract", "body"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

def write_output(rows, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows, columns=["paper_id", "gen_abstract"]).to_csv(out_path, index=False, encoding="utf-8")

def make_run_logger():
    os.makedirs("runs", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"runs/{stamp}.log"
    return log_path, open(log_path, "w", encoding="utf-8")
