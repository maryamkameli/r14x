# The 32-row validation subset the smoke test uses (smoke.sbatch: TEST_FILE=$B/valid32.parquet).
import sys
import pandas as pd
src, out = sys.argv[1], sys.argv[2]   # <ampo_official>/data/valid.parquet  <baselines>/valid32.parquet
pd.read_parquet(src).sample(32, random_state=0).to_parquet(out)
