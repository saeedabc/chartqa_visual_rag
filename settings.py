import os

if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = os.path.expanduser("~/scratch/.cache/huggingface")

ROOT_DIR = os.path.expanduser("~/scratch/td2_usecase/data")
os.makedirs(ROOT_DIR, exist_ok=True)

TEAM_ROOT_DIR = "/projects/multimodal_bootcamp/multimodal-td-2/shared"

RET_MODEL_PATHS = {
    "colpali13": "vidore/colpali-v1.3",
}

GEN_MODEL_PATHS = {
    # "qwen25_vl_7b_instruct": "/fs02/model-weights/Qwen2.5-VL-7B-Instruct",  # NOTE: ValueError: Unrecognized image processor in /fs02/model-weights/Qwen2.5-VL-7B-Instruct
    "qwen25_vl_7b_instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen25_vl_32b_instruct": "/fs02/model-weights/Qwen2.5-VL-32B-Instruct",
    "qwen25_vl_72b_instruct_awq": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
}