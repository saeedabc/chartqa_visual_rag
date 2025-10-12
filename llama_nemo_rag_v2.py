import os, json, math, pickle
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from PIL import Image as PILImage, ImageFile
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import AutoModel
import matplotlib.pyplot as plt
from numpy.lib.format import open_memmap
import gc

def _iter_image_paths(img_dir: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(
        [os.path.join(img_dir, f) for f in os.listdir(img_dir)
         if os.path.splitext(f.lower())[1] in exts]
    )

def _load_images(paths: List[str]) -> List[PILImage.Image]:
    imgs = []
    for p in paths:
        img = PILImage.open(p).convert("RGB")
        imgs.append(img)
    return imgs

def _batched(iterable, bs: int):
    for i in range(0, len(iterable), bs):
        yield iterable[i:i+bs]

def _cosine_normalize(x: np.ndarray) -> np.ndarray:
    # x: [N, D]
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


@dataclass
class LlamaNemoIndex:
    model_name: str
    dim: int
    ids: List[int]
    image_paths: List[str]
    metadata: List[Dict]
    embeddings: np.ndarray  

    def save(self, index_root: str, index_name: str):
        os.makedirs(index_root, exist_ok=True)
        base = os.path.join(index_root, index_name)
        np.save(f"{base}.embeddings.npy", self.embeddings.astype(np.float32))
        with open(f"{base}.meta.pkl", "wb") as f:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "dim": self.dim,
                    "ids": self.ids,
                    "image_paths": self.image_paths,
                    "metadata": self.metadata,
                },
                f,
            )

    @staticmethod
    def load(index_root: str, index_name: str) -> "LlamaNemoIndex":
        base = os.path.join(index_root, index_name)
        embeddings = np.load(f"{base}.embeddings.npy", mmap_mode="r")
        with open(f"{base}.meta.pkl", "rb") as f:
            meta = pickle.load(f)
        return LlamaNemoIndex(
            model_name=meta["model_name"],
            dim=int(embeddings.shape[1]),
            ids=meta["ids"],
            image_paths=meta["image_paths"],
            metadata=meta["metadata"],
            embeddings=embeddings,
        )

# Make it Byaldi-like retriever

class RAGMultiModalModel:

    def __init__(self, model: AutoModel, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.eval()

        if not all(
            hasattr(self.model, attr) for attr in
            ["forward_passages", "forward_queries", "get_scores"]
        ):
            raise AttributeError(
                "The loaded model doesn't expose forward_passages/forward_queries/get_scores"
            )

        # probe dims once
        with torch.no_grad():
            qe = self.model.forward_queries(["probe"]).detach().cpu().numpy()
        self.dim = int(qe.shape[-1])

        self._index: Optional[LlamaNemoIndex] = None

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "nvidia/llama-nemoretriever-colembed-3b-v1",
        device_map: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        **hf_kwargs,
    ) -> "RAGMultiModalModel":
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=hf_kwargs.pop("attn_implementation", "flash_attention_2"),
            **hf_kwargs,
        )
        return cls(model=model, device="cuda" if "cuda" in str(device_map) else "cpu")

    @classmethod
    def from_index(
        cls,
        index_path: str,
        index_root: Optional[str] = None,
        model_name: Optional[str] = None,
        **hf_kwargs,
    ) -> "RAGMultiModalModel":
        if index_root is None:
            # index_path is a full "root/name" without extension
            root = os.path.dirname(index_path)
            name = os.path.basename(index_path)
        else:
            root, name = index_root, index_path

        idx = LlamaNemoIndex.load(root, name)
        # If caller didn't pass a model_name, reuse the one stored in index
        model_name = model_name or idx.model_name
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            **hf_kwargs,
        )
        inst = cls(model=model, device="cuda")
        inst._index = idx
        return inst

    def index(
        self,
        input_path: str,
        index_name: str,
        index_root: str,
        metadata: list[dict] | None = None,
        overwrite: bool = False,
        batch_size: int = 4,      # safe default
        max_side: int = 512,      # tighten tokens; bump to 768 if stable
        store_collection_with_index: bool = False,
    ) -> None:

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        os.makedirs(index_root, exist_ok=True)
        base = os.path.join(index_root, os.path.basename(index_name))
    
        # overwrite guard
        if (os.path.exists(f"{base}.embeddings.npy") or os.path.exists(f"{base}.meta.pkl")) and not overwrite:
            raise FileExistsError(f"Index already exists at {base}.* (use overwrite=True)")
    
        # collect images
        img_dir = os.path.abspath(os.path.expanduser(input_path))
        img_paths = sorted(
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if os.path.splitext(f.lower())[1] in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
        if not img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
    
        if metadata is None:
            metadata = [{"image_name": os.path.basename(p)} for p in img_paths]
        else:
            assert len(metadata) == len(img_paths), "metadata length must equal number of images"
    
        # probe dimensionality after pooling
        with PILImage.open(img_paths[0]) as im0:
            im0 = im0.convert("RGB")
            im0.thumbnail((max_side, max_side), resample=PILImage.BICUBIC)
            with torch.no_grad():
                pe0 = self.model.eval().forward_passages([im0], batch_size=1)  # [1,T,D]
                pe0 = pe0.mean(dim=1)                                          # [1,D]
                D = int(pe0.shape[-1])
        N = len(img_paths)
    
        # disk-backed array with a proper .npy header
        embs_mm = open_memmap(f"{base}.embeddings.npy", mode="w+", dtype=np.float32, shape=(N, D))
    
        def _prep(p):
            with PILImage.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((max_side, max_side), resample=PILImage.BICUBIC)
                return im
    
        i = 0
        with torch.no_grad():
            while i < N:
                bs = min(batch_size, N - i)
                try:
                    imgs = [_prep(p) for p in img_paths[i:i+bs]]
                    pe = self.model.eval().forward_passages(imgs, batch_size=len(imgs))  # [B,T,D]
                    pe = pe.mean(dim=1).to(dtype=torch.float32)                           # [B,D]
                    embs_mm[i:i+bs, :] = pe.detach().cpu().numpy()
                    i += bs
                finally:
                    # always free between batches
                    del imgs
                    if 'pe' in locals():
                        del pe
                    torch.cuda.empty_cache()
                    gc.collect()
    
        # L2-normalize in place (cosine similarity ready)
        norms = np.linalg.norm(embs_mm, axis=1, keepdims=True) + 1e-12
        embs_mm[:] = embs_mm / norms
        del embs_mm  # flush to disk
    
        # save meta
        with open(f"{base}.meta.pkl", "wb") as f:
            pickle.dump(
                dict(
                    model_name=self.model.name_or_path,
                    dim=D,
                    ids=list(range(N)),
                    image_paths=img_paths,
                    metadata=metadata,
                ),
                f,
            )
    
        # ready to use immediately
        self._index = LlamaNemoIndex.load(index_root, os.path.basename(index_name))



    def search(self, query: str, k: int = 3) -> list[dict]:
        if self._index is None:
            raise RuntimeError("No index loaded. Build or load an index first.")
    
        import numpy as np
        import torch
    
        # 1) embed query -> [1, T, D], then pool to [D]
        with torch.no_grad():
            q = self.model.eval().forward_queries([query], batch_size=1)  # [1, T, D]
            q = q.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)  # [D]
        # 2) L2-normalize query (index rows already normalized during .index)
        q /= (np.linalg.norm(q) + 1e-12)
    
        # 3) cosine similarity via dot product
        sims = self._index.embeddings @ q  # [N]
    
        # 4) top-k
        k = min(k, sims.shape[0])
        topk_idx = np.argpartition(-sims, k-1)[:k]
        topk_idx = topk_idx[np.argsort(-sims[topk_idx])]
    
        # 5) format Byaldi-style results
        out = []
        for i in topk_idx:
            out.append({
                "doc_id": int(self._index.ids[i]),
                "page_num": 1,
                "score": float(sims[i] * 100.0),  # optional scaling
                "metadata": dict(self._index.metadata[i]),
                "base64": None,
            })
        return out


    def load_image(self, image_name: str, index_path: str) -> PILImage:
        image_path = f'{index_path}/{image_name}'
        image = PILImage.open(image_path)
        return image
    
    def plot_results(self, results: list[dict], index_path: str) -> None:
        k = len(results)
        fig, axes = plt.subplots(1, k, figsize=(4 * k, 10))  #, dpi=120)
    
        for res, ax in zip(results, axes.flat):
            img = self.load_image(res['metadata']['image_name'], index_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Score: {res['score']}")
            
        plt.tight_layout()
        plt.show()

