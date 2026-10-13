import sys
from huggingface_hub import HfApi

try:
    api = HfApi()
    files = api.list_repo_files(repo_id="MedOtter/brats2023-gli-dataset", repo_type="dataset")
    print("Total files in MedOtter/brats2023-gli-dataset:", len(files))
    for f in files[:20]:
        print("  -", f)
except Exception as e:
    print("Error querying HuggingFace:", e)
