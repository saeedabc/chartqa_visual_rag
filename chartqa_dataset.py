from datasets import load_dataset, Image, DatasetDict, Dataset
import hashlib


def load_chartqa_dataset(split: str = None) -> DatasetDict | Dataset:
    
    # Load dataset
    dset = load_dataset("HuggingFaceM4/ChartQA", split=split)

    # Cast images to "bytes"
    dset = dset.cast_column("image", Image(decode=False))

    # Hash image bytes
    def get_unique_id(row):
        h = hashlib.md5(row["image"]["bytes"]).hexdigest()
        return {"image_id": f"img_{h}"}

    dset = dset.map(get_unique_id, desc="Hashing image bytes to ids")

    # Cast back to images
    dset = dset.cast_column("image", Image(decode=True))
    
    return dset
