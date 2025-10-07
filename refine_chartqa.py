import json
import argparse
from tqdm.auto import tqdm
from datasets import Dataset

from settings import *
from chartqa_dataset import load_chartqa_dataset
from multi_modal import MultiModalGenerator


SYSTEM_PROMPT = """Your task is to refine a question that references an image of a chart.
The original question is highly image-dependent, so you should reframe it into a globally understandable question that is clear and well-defined without the image.
- Preserve the original intent of the question. 
- Add enough context from the chart so the question stands alone.
- Do not refer to the image and do not hint the answer in your refined question. 
- Provide a consistent refined answer to the refined question.
- Keep the original answer unless the question intent has to change.
Finally, output the refined question and answer in JSON format:
{{
  "question": "XXXXXX",
  "answer": ["YYYYYY"]
}}
Where "question" is your refined question, and "answer" is a list containing the answer. Do not output anything else.
"""

PROMPT = """Refine the given question and answer with respect to the given image.

Input Question:
{question}

Input Answer:
{answer}

Your Refined Question and Answer:
"""

def _parse_result(result: str) -> dict:
    if not result or result.isspace():
        return {}
    
    json_str = result.strip()
    
    if json_str.startswith('```json'):
        json_str = json_str[7:]  # Remove ```json
    if json_str.startswith('```'):
        json_str = json_str[3:]   # Remove ```
    if json_str.endswith('```'):
        json_str = json_str[:-3]  # Remove trailing ```
    
    json_str = json_str.strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw string: {json_str}")
        return {}

def refine_chartqa(
    vlm_name: str, 
    split: str = "val", 
    max_samples: int = None, 
    force_update: bool = False,
) -> Dataset:
    
    def generate_refined_qa(row: dict, vlm: MultiModalGenerator) -> dict:
        image = row['image']
        query = PROMPT.format(question=row['query'], answer=row['label'])
    
        result = vlm.generate(
            system_prompt=SYSTEM_PROMPT,
            query=query,
            images=[image],
            max_new_tokens=128,
            output_parser=_parse_result, 
            verbose=False
        )

        return {
            f'refined_query': result.get('question'),
            f'refined_label': result.get('answer')
        }

    def refine_dset(dset: Dataset, vlm: MultiModalGenerator) -> Dataset:
        refined_rows = []
        for row in tqdm(dset, disable=False, desc="Generate Refined QAs"):
            refined_row = generate_refined_qa(row, vlm)
            refined_rows.append(refined_row)
    
        cols = refined_rows[0].keys()
        for col in cols:
            dset = dset.add_column(col, [row[col] for row in refined_rows])

        return dset

    # Refine QAs
    save_dir = f"{TEAM_ROOT_DIR}/refined_chartqa/{split}-{max_samples}_{vlm_name}"
    if force_update or not os.path.exists(save_dir):
        # Refine QAs and save to disk

        # Load dataset    
        dset = load_chartqa_dataset(split=split)
        if max_samples is not None:
            # dset = dset.select(range(max_samples))
            dset = dset.take(max_samples)

        # Load VLM
        vlm = MultiModalGenerator(model_name=GEN_MODEL_PATHS[vlm_name])

        # dset = dset.map(generate_refined_qa, batched=False, num_proc=1, desc=f"Generate Refined QAs with `{vlm_name}`")
        dset = refine_dset(dset, vlm)

        print(f"Saving the refined chartqa dataset to {save_dir}")
        dset.save_to_disk(save_dir)
    else:
        # Load refined QAs from disk
        print(f"Loading the refined chartqa dataset from {save_dir}")
        dset = Dataset.load_from_disk(save_dir)
    
    return dset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to refine ChartQA for a VRAG use case")
    parser.add_argument("--vlm_name", type=str, required=True, help="Name of the VLM model.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to use (e.g., train, val, test). Default: val")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process. Default: None")
    parser.add_argument("--force_update", action="store_true", help="Overwrite the existing refined dataset if exists. Default: False")
    args = parser.parse_args()
        
    refined_dset = refine_chartqa(vlm_name=args.vlm_name,
                                  split=args.split, 
                                  max_samples=args.max_samples,
                                  force_update=args.force_update) 
