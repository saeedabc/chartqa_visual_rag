import os

if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = os.path.expanduser("~/scratch/.cache/huggingface")

ROOT_DIR = os.path.expanduser("~/scratch/td2_usecase/data")
os.makedirs(ROOT_DIR, exist_ok=True)