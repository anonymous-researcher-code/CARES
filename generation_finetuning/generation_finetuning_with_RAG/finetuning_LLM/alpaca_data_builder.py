import json
import os
from tqdm import tqdm
from typing import List, Dict, Optional


class PathConfig:
    DATASET_BASE = "../dataset_path/"
    TRAIN_FILE = os.path.join(DATASET_BASE, "train.jsonl")
    VAL_FILE = os.path.join(DATASET_BASE, "valid.jsonl")
    TEST_FILE = os.path.join(DATASET_BASE, "test.jsonl")
    RETRIEVAL_BASE = "../rag_candidates"
     OUTPUT_DIR = os.path.join(DATASET_BASE, "../your_output_path/")
    TOP_K = 4 
    TOTAL_TOPK = 30

    USE_RAG = True  # Whether to use RAG (False means vanilla mode)
    RAG_FORMAT = "full_listing"  # RAG pair insertion format: "full_listing"
    

path_config = PathConfig()


class RAGContextFormatter:  
    @staticmethod
    def format_full_listing(candidate_codes: List[str], 
                           candidate_comments: List[str]) -> str:
        """
        :param candidate_codes: Retrieved code diff list
        :param candidate_comments: Retrieved review comment list
        :return: Formatted context string
        """
        if not candidate_codes or not candidate_comments:
            return ""
        
        context_blocks = []
        for i, (code, comment) in enumerate(zip(candidate_codes, candidate_comments), 1):
            block = f"""Example {i}:
Related Code Diff:
{code}

Related Review Comment:
{comment}"""
            context_blocks.append(block)
        
        # Use '---' to separate each example
        return "\n\n" + "-"*20 + "\n\n----------\n\n".join(context_blocks) + "\n\n" + "-"*20 


def load_dataset(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file does not exist: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]
    return dataset

def get_topk_candidates(topk: int, total_topk: int, file_path: str) -> Optional[List[List[str]]]:
    """
    Get top-k candidates from a retrieval result file.
    :param topk: Number of candidates needed
    :param total_topk: Total number of candidates per instance in the file
    :param file_path: Retrieval result file path
    :return: Top-k candidate list per instance, or None if file does not exist
    """
    if not os.path.exists(file_path):
        print(f"Warning: Retrieval result file does not exist: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        file_lines = f.read().splitlines()
    
    num_instances = len(file_lines) // total_topk
    retrieved = [
        file_lines[i * total_topk:(i + 1) * total_topk][:topk] 
        for i in range(num_instances)
    ]

    return retrieved

# =====================================================================
# RAG-Alpaca dataset builder
# =====================================================================
def build_rag_alpaca_dataset(
    dataset: List[Dict],
    candidate_codes: Optional[List[List[str]]],
    candidate_comments: Optional[List[List[str]]],
    output_path: str,
    rag_format_method: str = "full_listing",
    use_rag: bool = True
):
    formatter = RAGContextFormatter()
    format_functions = {
        "full_listing": formatter.format_full_listing
    }

    
    format_func = format_functions[rag_format_method]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(len(dataset)), desc=f"Building Alpaca dataset"):
            data = dataset[i]
            code_diff = data.get("diff", "")
            target_comment = data.get("msg", "")

            if use_rag and candidate_codes and candidate_comments and i < len(candidate_codes):
                # === RAG mode ===
                rag_context = format_func(candidate_codes[i], candidate_comments[i])
                instruction = (
                    "You are an expert code reviewer. Your task is to write a professional and concise review comment about the target code diff. "
                    "You have access to a list of Reference Exemplars from the historical codebase."
                    "You may refer to the retrieved similar examples for hints, patterns, or coding standards, but focus on analyzing the target diff itself."
                    "WARNING: Some retrieved examples might be irrelevant. If an example does not apply to the current Diff, ignore it."
                ) 
                
                input_text = f"""Retrieved Similar Examples:
{rag_context}

---

Target Code Diff to Review:
{code_diff}"""
            
            else:
                # === Vanilla mode (without RAG) ===
                instruction = (
                    "You are an expert code reviewer. Your task is to write a professional and concise review comment about the provided code diff."
                )
                input_text = code_diff
            
            alpaca_entry = {
                "instruction": instruction,
                "input": input_text,
                "output": target_comment
            }
            
            fout.write(json.dumps(alpaca_entry, ensure_ascii=False) + "\n")
    
    print(f"✓ Dataset saved to: {output_path}")

# =====================================================================
# Main pipeline
# =====================================================================
def main():
    USE_RAG = path_config.USE_RAG
    RAG_FORMAT = path_config.RAG_FORMAT
    TOP_K = path_config.TOP_K

    print(f"\nConfiguration:")
    print(f"  - Use RAG: {USE_RAG}")
    print(f"  - RAG format: {RAG_FORMAT}")
    print(f"  - Top-K: {TOP_K}")
    
    # Step 1: Load raw datasets
    print(f"\n{'='*10}")
    print("[Step 1/3] Loading raw datasets...")
    print(f"{'='*10}")
    
    train_dataset = load_dataset(path_config.TRAIN_FILE)
    val_dataset = load_dataset(path_config.VAL_FILE)
    test_dataset = load_dataset(path_config.TEST_FILE)
    
    # Step 2: Load RAG retrieval candidates (if RAG is enabled)
    print(f"\n{'='*10}")
    print("[Step 2/3] Loading RAG retrieval candidates...")
    print(f"{'='*10}")
    
    if USE_RAG:
        retrieval_base = path_config.RETRIEVAL_BASE
        
        print("\nTraining set retrieval results:")
        train_codes = get_topk_candidates(
            path_config.TOP_K, 
            path_config.TOTAL_TOPK,
            os.path.join(retrieval_base, "train_to_train_retrieval_top30_code.txt")
        )
        train_comments = get_topk_candidates(
            path_config.TOP_K, 
            path_config.TOTAL_TOPK,
            os.path.join(retrieval_base, "train_to_train_retrieval_top30_comment.txt")
        )
        
        print("\nValidation set retrieval results:")
        val_codes = get_topk_candidates(
            path_config.TOP_K, 
            path_config.TOTAL_TOPK,
            os.path.join(retrieval_base, "val_to_train_retrieval_top30_code.txt")
        )
        val_comments = get_topk_candidates(
            path_config.TOP_K, 
            path_config.TOTAL_TOPK,
            os.path.join(retrieval_base, "val_to_train_retrieval_top30_comment.txt")
        )
        
        print("\nTest set retrieval results:")
        test_codes = get_topk_candidates(
            path_config.TOP_K, 
            path_config.TOTAL_TOPK,
            os.path.join(retrieval_base, "test_to_train_retrieval_top30_code.txt")
        )
        test_comments = get_topk_candidates(
            path_config.TOP_K, 
            path_config.TOTAL_TOPK,
            os.path.join(retrieval_base, "test_to_train_retrieval_top30_comment.txt")
        )
    else:
        print("Skipping RAG retrieval (Vanilla mode)")
        train_codes = train_comments = None
        val_codes = val_comments = None
        test_codes = test_comments = None
    
    # Step 3: Build RAG-Alpaca datasets
    print(f"\n{'='*10}")
    print("[Step 3/3] Building RAG-Alpaca datasets...")
    print(f"{'='*10}")
    
    output_dir = path_config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)

    print("\n[Training Set]")
    build_rag_alpaca_dataset(
        dataset=train_dataset,
        candidate_codes=train_codes,
        candidate_comments=train_comments,
        output_path=os.path.join(output_dir, f"train_ragpair{TOP_K}_{RAG_FORMAT}_alpaca.jsonl"),
        rag_format_method=RAG_FORMAT,
        use_rag=USE_RAG
    )

    print("\n[Validation Set]")
    build_rag_alpaca_dataset(
        dataset=val_dataset,
        candidate_codes=val_codes,
        candidate_comments=val_comments,
        output_path=os.path.join(output_dir, f"valid_ragpair{TOP_K}_{RAG_FORMAT}_alpaca.jsonl"),
        rag_format_method=RAG_FORMAT,
        instruction_template=INSTRUCTION_TEMPLATE,
        use_rag=USE_RAG
    )
    
    print("\n[Test Set]")
    build_rag_alpaca_dataset(
        dataset=test_dataset,
        candidate_codes=test_codes,
        candidate_comments=test_comments,
        output_path=os.path.join(output_dir, f"test_ragpair{TOP_K}_{RAG_FORMAT}_alpaca.jsonl"),
        rag_format_method=RAG_FORMAT,
        use_rag=USE_RAG
    )


if __name__ == "__main__":
    main()