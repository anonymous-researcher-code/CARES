import os
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json

def load_dataset(base_path):
    paths = {
        "train": os.path.join(base_path, "msg-train.jsonl"),
        "val": os.path.join(base_path, "msg-valid.jsonl"),
        "test": os.path.join(base_path, "msg-test.jsonl")
    }
    datasets = {}
    for split, path in paths.items():
        with open(path, "r", encoding="utf-8") as f:
            # Read jsonl line by line, each line is a JSON object
            datasets[split] = [json.loads(line) for line in f]
    return datasets

def process_dataset(dataset):
    code_list, comment_list = [], []
    for data in dataset:
        data = json.loads(data)
        code = data.get('new', '')
        comment = data.get('msg', '')
        code_list.append(code)
        comment_list.append(comment)
    return code_list, comment_list

def get_embeddings(texts, tokenizer, model, device, batch_size=16):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(**encoded)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                batch_emb = outputs.pooler_output
            else:
                last_hidden_state = outputs.last_hidden_state
                input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
                sum_mask = input_mask_expanded.sum(dim=1)
                batch_emb = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
            embeddings.append(batch_emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main():
    # === Setup ===
    base_path = "../dataset_base_path/"
    embedding_dir = "../code_embeddings"
    os.makedirs(embedding_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === Load model ===
    checkpoint = "microsoft/unixcoder-base-nine"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device)

    # === Load and process datasets ===
    datasets = load_dataset(base_path)
    train_code, _ = process_dataset(datasets["train"])
    val_code, _ = process_dataset(datasets["val"])
    test_code, _ = process_dataset(datasets["test"])

    # === Encode ===
    train_emb = get_embeddings(train_code, tokenizer, model, device)
    val_emb = get_embeddings(val_code, tokenizer, model, device)
    test_emb = get_embeddings(test_code, tokenizer, model, device)

    === Save ===
    with open(os.path.join(embedding_dir, "train_code_embeddings.pkl"), "wb") as f:
        pickle.dump(train_emb, f)
    with open(os.path.join(embedding_dir, "val_code_embeddings.pkl"), "wb") as f:
        pickle.dump(val_emb, f)
    with open(os.path.join(embedding_dir, "test_code_embeddings.pkl"), "wb") as f:
        pickle.dump(test_emb, f)


    print("Embeddings saved successfully.")
    print("Train:", train_emb.shape, "Val:", val_emb.shape, "Test:", test_emb.shape)

if __name__ == "__main__":
    main()
