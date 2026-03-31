"""
Workflow:
1. Load the trained model.
2. For each sample in the test set, retrieve top-k candidates from the original data.
3. Save candidates to files (every k lines form one group).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import json
import numpy as np
from typing import List, Dict, Tuple
import os


class Retrieval:  
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Args:
            model: Trained UniXcoderContrastiveModel.
            tokenizer: HuggingFace tokenizer
            device: Device.
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts (format: diff).
            batch_size: Batch size.
        
        Returns:
            embeddings (n, 256)
        """
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"  Encoding {len(texts)} texts...")
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_num = i // batch_size + 1
                if batch_num % 50 == 0 or batch_num == 1 or batch_num == total_batches:
                    print(f"    Batch {batch_num}/{total_batches}")
                
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Encode
                batch_emb = self.model.encode(input_ids, attention_mask)
                embeddings.append(batch_emb.cpu().numpy())
        
        print(f"  ✓ Encoding completed")
        return np.vstack(embeddings)
    
    def retrieve_and_save(
        self,
        original_data_file: str,
        test_data_file: str,
        output_code_file: str,
        output_comment_file: str,
        top_k: int = 30
    ):
        """
        Retrieve and save results.
        
        Args:
            original_data_file: Original full database file.
            test_data_file: Test data file.
            output_code_file: Output code file.
            output_comment_file: Output comment file.
            top_k: Retrieve top-k candidates.
        """
        print("="*70)
        print("Retrieval and Saving")
        print("="*70)
        
        # 1. Load data
        print("\n[1/5] Loading data...")
        with open(original_data_file, 'r', encoding='utf-8') as f:
            original_data = [json.loads(line) for line in f]
        print(f"  Original data: {len(original_data)} items")
        
        with open(test_data_file, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]
        print(f"  Test data: {len(test_data)} items")
        
        # 2. Encode original data
        print("\n[2/5] Encoding original data (building retrieval database)...")
        original_texts = [
            f"{item['diff']}"
            for item in original_data
        ]
        original_embeddings = self.encode_texts(original_texts)
        
        # Normalize (for cosine similarity)
        print("  Normalizing embeddings...")
        norms = np.linalg.norm(original_embeddings, axis=1, keepdims=True)
        original_embeddings = original_embeddings / (norms + 1e-8)
        
        # 3. Encode test data
        print("\n[3/5] Encoding test data...")
        test_texts = [
            f"{item['diff']}"
            for item in test_data
        ]
        test_embeddings = self.encode_texts(test_texts)
        
        # Normalize
        norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / (norms + 1e-8)
        
        # 4. Retrieve
        print(f"\n[4/5] Retrieving top-{top_k} candidates...")
        
        # Open output files
        f_code = open(output_code_file, 'w', encoding='utf-8')
        f_comment = open(output_comment_file, 'w', encoding='utf-8')
        
        all_similarities = []  # For later metric calculation
        
        for test_idx in range(len(test_data)):
            if (test_idx + 1) % 1000 == 0 or test_idx == 0:
                print(f"  Progress: {test_idx + 1}/{len(test_data)}")
            
            # Compute similarity with all original data
            query_emb = test_embeddings[test_idx:test_idx+1]
            similarities = np.dot(original_embeddings, query_emb.T).flatten()
            
            # Get top-k indices (excluding itself)
            top_indices = np.argsort(similarities)[::-1]
            
            # Filter out itself (if the test sample is also in original data)
            filtered_indices = []
            for idx in top_indices:
                # 检查是否是自己
                if (original_data[idx]['diff'] == test_data[test_idx]['diff'] and
                    original_data[idx]['msg'] == test_data[test_idx]['msg']):
                    continue
                filtered_indices.append(idx)
                if len(filtered_indices) >= top_k:
                    break
            
            # If fewer than top_k, fill with remaining candidates
            if len(filtered_indices) < top_k:
                for idx in top_indices:
                    if idx not in filtered_indices:
                        filtered_indices.append(idx)
                    if len(filtered_indices) >= top_k:
                        break
            
            # Save candidates
            candidate_sims = []
            for rank, idx in enumerate(filtered_indices[:top_k]):
                # Save diff code
                f_code.write(json.dumps(original_data[idx]['diff'], ensure_ascii=False) + '\n')
                
                # Save comment
                f_comment.write(original_data[idx]['msg'] + '\n')
            
        
        f_code.close()
        f_comment.close()
        
        print(f"\n  ✓ Candidate saving completed")
        print(f"    Code: {output_code_file}")
        print(f"    Comment: {output_comment_file}")
        print(f"    Format: Every {top_k} lines correspond to one test sample")


def run_retrieve(
    original_data_file: str,
    test_data_file: str,
    checkpoint_path: str,
    output_code_file: str,
    output_comment_file: str,
    top_k: int = 30,
):
    
    # Check files
    if not os.path.exists(original_data_file):
        print(f"Error: Original data not found: {original_data_file}")
        return
    
    if not os.path.exists(test_data_file):
        print(f"Error: Test data not found: {test_data_file}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found: {checkpoint_path}")
        return
    
    # Load model
    print("\nLoading model...")
    from transformers import AutoTokenizer
    from unixcoder_contrastive_trainer import UniXcoderContrastiveModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
    model = UniXcoderContrastiveModel(
        model_name="microsoft/unixcoder-base-nine",
        projection_dim=256
    )
    
    # Load weights
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Model loaded successfully")
    print(f"    Epoch: {checkpoint['epoch']}")
    print(f"    Val Loss: {checkpoint['val_loss']:.4f}")
    
    
    Retriever = Retrieval(model, tokenizer, device)
    
    # Retrieve and save
    metrics = Retriever.retrieve_and_save(
        original_data_file=original_data_file,
        test_data_file=test_data_file,
        output_code_file=output_code_file,
        output_comment_file=output_comment_file,
        top_k=top_k
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        top_k = int(sys.argv[1])
    else:
        top_k = 30
    
    print(f"Top-K: {top_k}")

    
    run_retrieve(
        original_data_file="your_full_database_path",
        test_data_file="../train.jsonl",
        checkpoint_path='./checkpoints/best_model.pt',
        output_code_file='../train_to_train_retrieval_top30_code.txt',
        output_comment_file='../train_to_train_retrieval_top30_comment.txt',
        top_k=top_k
    )
    run_retrieve(
        original_data_file="your_full_database_path",
        test_data_file="../test.jsonl",
        checkpoint_path='./checkpoints/best_model.pt',
        output_code_file='../test_to_train_retrieval_top30_code.txt',
        output_comment_file='../test_to_train_retrieval_top30_comment.txt',
        top_k=top_k
    )
    run_retrieve(
        original_data_file="your_full_database_path",
        test_data_file="../valid.jsonl",
        checkpoint_path='./checkpoints/best_model.pt',
        output_code_file='../val_to_train_retrieval_top30_code.txt',
        output_comment_file='../val_to_train_retrieval_top30_comment.txt',
        top_k=top_k
    )