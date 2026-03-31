import os
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json

def load_dataset(base_path):
    paths = {
        "train": os.path.join(base_path, "train.jsonl"),
        "val": os.path.join(base_path, "valid.jsonl"),
        "test": os.path.join(base_path, "test.jsonl")
    }
    datasets = {}
    for split, path in paths.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                datasets[split] = [json.loads(line) for line in f]
        else:
            print(f"Warning: {path} not found.")
            datasets[split] = []
    return datasets

def process_dataset(dataset):
    code_list = []
    for data in dataset:
        if isinstance(data, str): 
            data = json.loads(data)
        code = data.get('diff', '')
        code_list.append(code)
    return code_list

def get_embeddings(texts, tokenizer, model, device, batch_size=64):
    """
    针对 CodeT5+ 110m 优化的编码函数。
    该模型直接输出 embedding
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            
            # 1. Tokenize
            encoded = tokenizer(batch, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
            
            # 2. Model Forward
            outputs = model(**encoded)
            
            # 3. Normalization
            batch_emb = torch.nn.functional.normalize(outputs, p=2, dim=1)
            
            embeddings.append(batch_emb.cpu().numpy())
            
    if len(embeddings) == 0:
        return np.array([])
    
    return np.concatenate(embeddings, axis=0)


def main():
    # === Setup ===
    base_path = "../dataset_base_path/"
    embedding_dir = "../code_embeddings"
    os.makedirs(embedding_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # === Load model===
    checkpoint = "Salesforce/codet5p-110m-embedding"
    print(f"Loading model: {checkpoint} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
    model.max_seq_length = 512

    datasets = load_dataset(base_path)
    
    if "train" in datasets:
        train_code = process_dataset(datasets["train"])
        
        # === Encode ===
        print(f"Encoding {len(train_code)} train codes...")
        train_code_emb = get_embeddings(train_code, tokenizer, model, device, batch_size=64) # 显存够大可以调大 batch_size

        # === Save ===
        save_path = os.path.join(embedding_dir, "train_code_embeddings.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(train_code_emb, f)
        print(f"Saved to {save_path}")

    if "val" in datasets:
        val_code = process_dataset(datasets["val"])
        
        # === Encode ===
        print(f"Encoding {len(val_code)} val codes...")
        val_code_emb = get_embeddings(val_code, tokenizer, model, device, batch_size=64) # 显存够大可以调大 batch_size

        # === Save ===
        save_path = os.path.join(embedding_dir, "val_code_embeddings.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(val_code_emb, f)
        print(f"Saved to {save_path}")

    if "test" in datasets:
        test_code = process_dataset(datasets["test"])
        
        # === Encode ===
        print(f"Encoding {len(test_code)} test codes...")
        test_code_emb = get_embeddings(test_code, tokenizer, model, device, batch_size=64) 

        # === Save ===
        save_path = os.path.join(embedding_dir, "test_code_embeddings.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(test_code_emb, f)
        print(f"Saved to {save_path}")

    print("All embeddings process finished.")

if __name__ == "__main__":
    main()