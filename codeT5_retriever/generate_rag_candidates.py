import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json

def processDataset(dataset):
    code_list, comment_list = [], []
    for data in dataset:
        code = data.get('diff', '')
        comment = data.get('msg', '')
        code_list.append(code)
        comment_list.append(comment)
    return code_list, comment_list

def getTopk_from_batch(batch_source_embeddings, target_embeddings, topk, skip_offset=None):
    similarity_matrix = cosine_similarity(batch_source_embeddings, target_embeddings)
    topk_indices_all = []

    for i, similarities in enumerate(similarity_matrix):
        if skip_offset is not None:
            similarities[skip_offset + i] = -1e10  # Exclude self-similarity
        topk_indices = np.argsort(similarities)[::-1][:topk]
        topk_indices_all.append(topk_indices.tolist())

    return topk_indices_all

def process_data_with_topk_candidates(data_embeddings, train_embeddings, train_code, train_comment,
                                      output_path_code, output_path_comment, topk, skip_itself=False, batch_size=10000):
    data_len = len(data_embeddings)

    with open(output_path_code, 'w', encoding="utf-8") as f_code, \
         open(output_path_comment, 'w', encoding="utf-8") as f_comment:

        for start_idx in tqdm(range(0, data_len, batch_size), desc="Processing in Batches"):
            end_idx = min(start_idx + batch_size, data_len)
            batch_embeddings = data_embeddings[start_idx:end_idx]

            topk_indices_batch = getTopk_from_batch(
                batch_embeddings,
                train_embeddings,
                topk,
                skip_offset=start_idx if skip_itself else None
            )

            for indices in topk_indices_batch:
                for idx in indices:
                    f_code.write(json.dumps(train_code[idx], ensure_ascii=False) + '\n')
                    f_comment.write(json.dumps(train_comment[idx], ensure_ascii=False) + '\n')

    print(f"Top-{topk} retrieval complete. Output saved to:\n- {output_path_code}\n- {output_path_comment}")

def run_topk_retrieval_for_split(split_name, data_embeddings, train_embeddings, train_code, train_comment,
                                 output_dir, topk=30, skip_itself=False):
    os.makedirs(output_dir, exist_ok=True)

    output_code_path = os.path.join(output_dir, f"{split_name}_to_train_retrieval_top{topk}_code.txt")
    output_comment_path = os.path.join(output_dir, f"{split_name}_to_train_retrieval_top{topk}_comment.txt")

    process_data_with_topk_candidates(
        data_embeddings,
        train_embeddings,
        train_code,
        train_comment,
        output_code_path,
        output_comment_path,
        topk,
        skip_itself
    )

def main():
    base_dir = "../dataset_path"
    embedding_dir = "../code_embeddings"
    output_dir = '../rag_comment_candidate'
    topk = 30
    
    train_path = os.path.join(base_dir, "train.jsonl")
    with open(train_path, "r", encoding="utf-8") as f:
        # 逐行读取jsonl，每行为一个json对象
        train_data = [json.loads(line) for line in f]
    train_code, train_comment = processDataset(train_data)

    # Load embeddings
    with open(os.path.join(embedding_dir, "train_code_embeddings.pkl"), "rb") as f:
        train_embeddings = pickle.load(f)
    with open(os.path.join(embedding_dir, "val_code_embeddings.pkl"), "rb") as f:
        val_embeddings = pickle.load(f)
    with open(os.path.join(embedding_dir, "test_code_embeddings.pkl"), "rb") as f:
        test_embeddings = pickle.load(f)

    # Run retrieval for each data split
    run_topk_retrieval_for_split("train", train_embeddings, train_embeddings, train_code, train_comment, output_dir, topk, skip_itself=True)
    run_topk_retrieval_for_split("val", val_embeddings, train_embeddings, train_code, train_comment, output_dir, topk, skip_itself=False)
    run_topk_retrieval_for_split("test", test_embeddings, train_embeddings, train_code, train_comment, output_dir, topk, skip_itself=False)

if __name__ == '__main__':
    main()
