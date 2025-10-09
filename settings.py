import os

_ROOT_DIR = "/projects/multimodal_bootcamp/multimodal-td-2"

USER = os.getenv("USER")
USER_ROOT_DIR = f"{_ROOT_DIR}/{USER}"

TEAM_ROOT_DIR = f"{_ROOT_DIR}/shared"

if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = os.path.expanduser(f"{USER_ROOT_DIR}/.cache/huggingface")
print(f'HF_HOME is set to {os.environ["HF_HOME"]}')

RET_MODEL_PATHS = {
    "colpali13": "vidore/colpali-v1.3",
    "colqwen2010": "vidore/colqwen2-v1.0",
    # "colqwen2502": "vidore/colqwen2.5-v0.2",
}

GEN_MODEL_PATHS = {
    # "qwen25_vl_7b_instruct": "/fs02/model-weights/Qwen2.5-VL-7B-Instruct",  # NOTE: ValueError: Unrecognized image processor in /fs02/model-weights/Qwen2.5-VL-7B-Instruct
    "qwen25_vl_7b_instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen25_vl_32b_instruct": "/fs02/model-weights/Qwen2.5-VL-32B-Instruct",
    "qwen25_vl_72b_instruct_awq": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
}