"""Print the schema and one sample of a codeforces-cots subset.

Run this before trusting data.py: the subsets do not share a column layout.

    pixi run python scripts/inspect_dataset.py solutions_py
"""

import sys

from datasets import load_dataset

subset = sys.argv[1] if len(sys.argv) > 1 else "solutions_py"
ds = load_dataset("open-r1/codeforces-cots", subset, split="train", streaming=True)

row = next(iter(ds))
print(f"subset: {subset}")
print(f"columns: {sorted(row)}\n")
for key, value in row.items():
    text = str(value)
    print(f"--- {key} ({type(value).__name__}, len={len(text)}) ---")
    print(text[:600])
    print()
