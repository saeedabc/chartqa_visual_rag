import os
from typing import Callable
import io
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset, Image, DatasetDict, Dataset
from PIL import Image as PILImage
from byaldi import RAGMultiModalModel

from chartqa_dataset import load_chartqa_dataset


def store_chartqa_images(split: str, save_dir: str) -> None:
    
    dset = load_chartqa_dataset(split=split)

    if not os.path.exists(save_dir):
        print(f'Storing images at {save_dir}')
        os.makedirs(save_dir)
        for row in tqdm(dset):
            img_path = f"{save_dir}/{row['image_id']}.png"
            img = row['image']
            img.save(img_path)
    else:
        print(f'Images already stored at {save_dir}')


def load_retriever(
    model_name: str, 
    imgs_dir: str, 
    index_root: str, 
    index_name: str
) -> RAGMultiModalModel:    

    index_path = f'{index_root}/{index_name}'
    
    if not os.path.exists(index_path):
        ret_model = RAGMultiModalModel.from_pretrained(model_name, index_root=index_root)
        # ret_model = RAGMultiModalModel.from_pretrained("vidore/colqwen2.5-v0.2")
    
        metadata = [{"image_name": f} for f in os.listdir(imgs_dir)]
        # img_ids = [m["filename"].rstrip('.png') for m in metadata]
        
        ret_model.index(
            input_path=imgs_dir,
            index_name=index_name,
            # doc_ids=img_ids,
            metadata=metadata,
            store_collection_with_index=False,
            overwrite=True
        )
    else:
        ret_model = RAGMultiModalModel.from_index(
            index_path=index_path,
            index_root=index_root,
        )

    return ret_model


class MultiModalRetriever:
    def __init__(
        self, 
        model_name: str = "vidore/colpali-v1.3", 
        split: str = "val", 
        root_dir: str | None = None
    ):
        if root_dir is None:
            root_dir = os.path.expanduser("~/scratch/td2_usecase/data")
        os.makedirs(root_dir, exist_ok=True)

        self.imgs_dir = f"{root_dir}/chartqa_images/{split}"
        store_chartqa_images(split=split, save_dir=self.imgs_dir)
        
        self.model = load_retriever(model_name,
                                    imgs_dir=self.imgs_dir,
                                    index_root=f"{root_dir}/.byaldi",
                                    index_name=f"chartqa_{split}_index")

    def search(
        self, 
        query: str, 
        k: int = 3, 
        verbose: bool = False
    ) -> list[dict]:
        ret_results = self.model.search(query, k=k)
        ret_results = [res.dict() for res in ret_results]
    
        # for res in ret_results:
        #     res['metadata']['image'] = self._load_image(res['metadata']['image_name'])

        if verbose:
            self.plot_results(ret_results)
        
        return ret_results

    def load_image(self, image_name: str) -> PILImage:
        image_path = f'{self.imgs_dir}/{image_name}'
        image = PILImage.open(image_path)
        return image
    
    def plot_results(self, ret_results: list[dict]) -> None:
        k = len(ret_results)
        fig, axes = plt.subplots(1, k, figsize=(4 * k, 10))  #, dpi=120)
    
        for res, ax in zip(ret_results, axes.flat):
            # img = res['metadata']['image']
            img = self.load_image(res['metadata']['image_name'])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Score: {res['score']}")
            
        plt.tight_layout()
        plt.show()


class MultiModalGenerator:
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    ):
        # Initialize VLM
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
        self.model.eval()

        # Initialize Processor
        min_pixels = 224*224
        max_pixels = 1024*1024
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )

    def generate(
        self, 
        system_prompt: str | None = None, 
        query: str | None = None, 
        images: list | None = None, 
        messages: list[dict] | None = None,
        max_new_tokens: int = 500,
        output_parser: Callable | None = None,
        verbose: bool = False,
    ) -> str:
        
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                })

            user_content = []
            if images:
                for image in images:
                    user_content.append({"type": "image", "image": image})
            if query:
                user_content.append({"type": "text", "text": query})
            if user_content:
                messages.append({
                    "role": "user",
                    "content": user_content,
                })

            if len(messages) == 0:
                raise ValueError("No input is provided")

        else:
            if system_prompt or query or images:
                raise ValueError("Provide either `messages` or any of `system_prompt`, `query`, and `images`, not both.")
    
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
    
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
    
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
        gen_result = output_text[0]

        if verbose:
            print(gen_result)

        if output_parser is not None:
            return output_parser(gen_result)
        
        return gen_result


class MultiModalRAG:
    def __init__(
        self, 
        retriever_args: dict | None = None,
        retriever: MultiModalRetriever | None = None,
        generator_args: dict = None,
        generator: MultiModalGenerator | None = None,
    ):
        self.retriever = retriever
        if self.retriever is None:
            self.retriever = MultiModalRetriever(**(retriever_args or {}))

        self.generator = generator
        if self.generator is None:
            self.generator = MultiModalGenerator(**(generator_args or {}))

    def run(
        self, 
        query: str, 
        k: int = 3, 
        verbose: bool = False
    ) -> tuple[list[dict], str]:
        
        # Retrieve
        ret_results = self.retriever.search(query=query,
                                            k=k,
                                            verbose=verbose)

        # Generate
        system_prompt = """You are a helpful assistant. Answer the given question based on the given chart(s)."""
        images = [self.retriever.load_image(res['metadata']['image_name']) for res in ret_results]
        gen_result = self.generator.generate(system_prompt=system_prompt,
                                             query=query,
                                             images=images,
                                             verbose=verbose)

        return ret_results, gen_result

    def _evaluate_retriever(self, ret_results: list[dict], gt_image_id: str) -> dict[str, float]:
        ret_image_ids = [res['metadata']['image_name'].rstrip('.png') for res in ret_results]
        recall = 1 if gt_image_id in ret_image_ids else 0
        return {'ret_recall': recall}

    def _evaluate_generator(self, query: str, gen_result: str, gt_answer: str) -> dict[str, float]:
        system_prompt = """You are a helpful evaluator.
Your task is to evaluate whether a model’s response to a given input question is correct, using the provided ground-truth label.

Rules:
- Output 1 if the response is correct based on the ground-truth.
- Output 0 otherwise.
- The ground-truth is usually a single word or number. The response should be considered correct if it is factual according to the ground-truth, regardless of formatting, capitalization, or minor syntactic differences.
- Do not output anything other than 1 or 0."""

        prompt = """Evaluate the correctness of model response.
        
Input query:
{query}

Model Response:
{gen_result}

Ground-truth Label:
{gt_answer}

Your binary 0-1 score:
"""

        def parse_result(result: str):
            try: 
                score = int(result.strip())
                assert score in [0, 1]
                return score
            except Error as e: 
                print(f"Parsing error: {e}")
                print(f"Raw string: {result}")
                return None
        
        correctness = self.generator.generate(system_prompt=system_prompt, 
                                              query=prompt.format(query=query, gen_result=gen_result, gt_answer=gt_answer),
                                              output_parser=parse_result,
                                              verbose=False)

        return {'gen_correctness': correctness}

    def evaluate(
        self, 
        query: str, 
        ret_results: list[dict], 
        gen_result: str, 
        gt_image_id: str | None = None, 
        gt_answer: str | None = None
    ) -> dict[str, float]:
        
        metrics = {}
        if gt_image_id is not None:
            metrics |= self._evaluate_retriever(ret_results, gt_image_id)
        
        if gt_answer is not None:
            metrics |= self._evaluate_generator(query, gen_result, gt_answer)

        return metrics
        
    def _aggregate_metrics(self, dset: Dataset) -> pd.DataFrame:
        def stats(values: list[float]) -> dict:
            return {
                'mean': np.nanmean(values),
                'std': np.nanstd(values),
            }

        cols = ['ret_recall', 'gen_correctness']
        metrics =  {col: stats(dset[col]) for col in dset.column_names if col in cols}
        return pd.DataFrame(metrics).round(4)
    
    def run_and_evaluate_all(
        self, 
        dset: Dataset,
        query_column: str = "qwen25_7b_query",
        gt_answer_column: str = "qwen25_7b_label",
        gt_image_id_column: str = "image_id",
        k: int = 3,
        save_dir: str | None = None,
    ) -> tuple[Dataset, pd.DataFrame]:
        
        # Load results from disk if exists
        if save_dir is not None and os.path.exists(save_dir):
            print(f"Loading RAG results from {save_dir}")
            dset = Dataset.load_from_disk(save_dir)
            return dset, self._aggregate_metrics(dset)
            
        new_columns = defaultdict(list)
    
        for row in tqdm(dset, desc="Visual RAG Inference and Evaluation"):
            query = row[query_column]
            gt_answer = row[gt_answer_column][0]
            gt_image_id = row[gt_image_id_column]

            # Run RAG
            ret_results, gen_result = self.run(query=query, 
                                               verbose=False)
            new_columns["ret_images"].append(ret_results)
            new_columns["gen_answer"].append(gen_result)

            # Evaluate
            metrics = self.evaluate(query=query, 
                                    ret_results=ret_results, 
                                    gen_result=gen_result, 
                                    gt_image_id=gt_image_id, 
                                    gt_answer=gt_answer)
            for metric, value in metrics.items():
                new_columns[metric].append(value)

        # Add new columns to the dataset
        for col in new_columns:
            dset = dset.add_column(col, new_columns[col])

        # Save results to disk
        if save_dir is not None:
            print(f"Saving RAG results to {save_dir}")
            dset.save_to_disk(save_dir)
        
        return dset, self._aggregate_metrics(dset)

