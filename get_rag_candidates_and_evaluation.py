"""
简单直接的检索评估

流程：
1. 加载训练好的模型
2. 对 test 集每条数据，从原始数据中检索 top-30 候选
3. 保存候选到文件（每30行为一组）
4. 计算检索指标（评估 comment 相似性）
"""
# 在运行检索脚本前，设置环境变量 KMP_DUPLICATE_LIB_OK=TRUE，允许重复加载：
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import json
import numpy as np
from typing import List, Dict, Tuple
import os
# from sentence_transformers import SentenceTransformer


class RetrievalEvaluator:
    """
    检索评估器
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Args:
            model: 训练好的 UniXcoderContrastiveModel
            tokenizer: HuggingFace tokenizer
            device: 设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
        # 用于评估 comment 相似度
        print("加载 comment 评估模型...")
        # self.comment_evaluator = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✓ 检索评估器初始化完成")
    
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        编码文本为 embedding
        
        Args:
            texts: 文本列表（格式：diff）
            batch_size: 批大小
        
        Returns:
            embeddings (n, 256)
        """
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"  编码 {len(texts)} 个文本...")
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_num = i // batch_size + 1
                if batch_num % 50 == 0 or batch_num == 1 or batch_num == total_batches:
                    print(f"    批次 {batch_num}/{total_batches}")
                
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
                
                # 编码
                batch_emb = self.model.encode(input_ids, attention_mask)
                embeddings.append(batch_emb.cpu().numpy())
        
        print(f"  ✓ 编码完成")
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
        检索并保存结果
        
        Args:
            original_data_file: 原始完整数据文件
            test_data_file: 测试数据文件
            output_code_file: 输出的 code 文件
            output_comment_file: 输出的 comment 文件
            top_k: 检索 top-k 个候选
        """
        print("="*70)
        print("检索与保存")
        print("="*70)
        
        # 1. 加载数据
        print("\n[1/5] 加载数据...")
        with open(original_data_file, 'r', encoding='utf-8') as f:
            original_data = [json.loads(line) for line in f]
        print(f"  原始数据: {len(original_data)} 条")
        
        with open(test_data_file, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]
        print(f"  测试数据: {len(test_data)} 条")
        
        # 2. 编码原始数据
        print("\n[2/5] 编码原始数据（构建检索库）...")
        original_texts = [
            f"{item['diff']}"
            for item in original_data
        ]
        original_embeddings = self.encode_texts(original_texts)
        
        # 归一化（用于余弦相似度）
        print("  归一化 embeddings...")
        norms = np.linalg.norm(original_embeddings, axis=1, keepdims=True)
        original_embeddings = original_embeddings / (norms + 1e-8)
        
        # 3. 编码测试数据
        print("\n[3/5] 编码测试数据...")
        test_texts = [
            f"{item['diff']}"
            for item in test_data
        ]
        test_embeddings = self.encode_texts(test_texts)
        
        # 归一化
        norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / (norms + 1e-8)
        
        # 4. 检索
        print(f"\n[4/5] 检索 top-{top_k} 候选...")
        
        # 打开输出文件
        f_code = open(output_code_file, 'w', encoding='utf-8')
        f_comment = open(output_comment_file, 'w', encoding='utf-8')
        
        all_similarities = []  # 用于后续计算指标
        
        for test_idx in range(len(test_data)):
            if (test_idx + 1) % 1000 == 0 or test_idx == 0:
                print(f"  进度: {test_idx + 1}/{len(test_data)}")
            
            # 计算与所有原始数据的相似度
            query_emb = test_embeddings[test_idx:test_idx+1]
            similarities = np.dot(original_embeddings, query_emb.T).flatten()
            
            # 获取 top-k 索引（不包括自己）
            top_indices = np.argsort(similarities)[::-1]
            
            # 过滤掉自己（如果测试样本也在原始数据中）
            filtered_indices = []
            for idx in top_indices:
                # 检查是否是自己
                if (original_data[idx]['diff'] == test_data[test_idx]['diff'] and
                    original_data[idx]['msg'] == test_data[test_idx]['msg']):
                    continue
                filtered_indices.append(idx)
                if len(filtered_indices) >= top_k:
                    break
            
            # 如果不够 top_k，补充剩余的
            if len(filtered_indices) < top_k:
                for idx in top_indices:
                    if idx not in filtered_indices:
                        filtered_indices.append(idx)
                    if len(filtered_indices) >= top_k:
                        break
            
            # 保存候选
            candidate_sims = []
            for rank, idx in enumerate(filtered_indices[:top_k]):
                # 保存 diff code
                f_code.write(json.dumps(original_data[idx]['diff'], ensure_ascii=False) + '\n')
                
                # 保存 comment
                f_comment.write(original_data[idx]['msg'] + '\n')
                
                # 记录相似度
                candidate_sims.append(similarities[idx])
            
            all_similarities.append(candidate_sims)
        
        f_code.close()
        f_comment.close()
        
        print(f"\n  ✓ 候选保存完成")
        print(f"    Code: {output_code_file}")
        print(f"    Comment: {output_comment_file}")
        print(f"    格式: 每 {top_k} 行对应一个测试样本")
        
        # 5. 计算评估指标
        # print(f"\n[5/5] 计算评估指标...")
        # metrics = self.compute_metrics(
        #     test_data,
        #     original_data,
        #     all_similarities,
        #     top_k
        # )
        
        return metrics if 'metrics' in locals() else None
    
    # def compute_metrics(
    #     self,
    #     test_data: List[Dict],
    #     original_data: List[Dict],
    #     all_similarities: List[List[float]],
    #     top_k: int
    # ) -> Dict:
    #     """
    #     计算评估指标
        
    #     使用 comment 语义相似度作为相关性判断
    #     """
    #     print("\n  计算评估指标...")
        
    #     # 编码所有 comments（用于评估）
    #     print("  编码 test comments...")
    #     test_comments = [item['comment'] for item in test_data]
    #     test_comment_embs = self.comment_evaluator.encode(
    #         test_comments,
    #         show_progress_bar=False,
    #         batch_size=128
    #     )
        
    #     print("  编码 original comments...")
    #     original_comments = [item['comment'] for item in original_data]
    #     original_comment_embs = self.comment_evaluator.encode(
    #         original_comments,
    #         show_progress_bar=False,
    #         batch_size=128
    #     )
        
    #     # 归一化
    #     test_comment_embs = test_comment_embs / (
    #         np.linalg.norm(test_comment_embs, axis=1, keepdims=True) + 1e-8
    #     )
    #     original_comment_embs = original_comment_embs / (
    #         np.linalg.norm(original_comment_embs, axis=1, keepdims=True) + 1e-8
    #     )
        
    #     # 计算指标
    #     print("  计算指标...")
        
    #     # 读取保存的候选
    #     with open('msg_test_to_train_retrieval_top30_comment.txt', 'r', encoding='utf-8') as f:
    #         all_retrieved_comments = f.readlines()
        
    #     metrics = {
    #         'precision_at_k': {k: [] for k in [1, 5, 10, 20, 30]},
    #         'recall_at_k': {k: [] for k in [1, 5, 10, 20, 30]},
    #         'mrr': [],
    #         'avg_similarity': []
    #     }
        
    #     similarity_threshold = 0.7  # comment 相似度阈值
        
    #     for test_idx in range(len(test_data)):
    #         if (test_idx + 1) % 1000 == 0:
    #             print(f"    进度: {test_idx + 1}/{len(test_data)}")
            
    #         # 获取该测试样本的候选
    #         start_line = test_idx * top_k
    #         retrieved_comments = all_retrieved_comments[start_line:start_line + top_k]
    #         retrieved_comments = [c.strip() for c in retrieved_comments]
            
    #         # 编码检索到的 comments
    #         retrieved_embs = self.comment_evaluator.encode(
    #             retrieved_comments,
    #             show_progress_bar=False
    #         )
    #         retrieved_embs = retrieved_embs / (
    #             np.linalg.norm(retrieved_embs, axis=1, keepdims=True) + 1e-8
    #         )
            
    #         # 计算与查询 comment 的相似度
    #         query_emb = test_comment_embs[test_idx:test_idx+1]
    #         comment_sims = np.dot(retrieved_embs, query_emb.T).flatten()
            
    #         # 判断相关性（comment 相似度超过阈值）
    #         relevant = comment_sims >= similarity_threshold
            
    #         # 记录平均相似度
    #         metrics['avg_similarity'].append(np.mean(comment_sims))
            
    #         # Precision@K 和 Recall@K
    #         for k in [1, 5, 10, 20, 30]:
    #             if k > len(relevant):
    #                 continue
                
    #             relevant_at_k = relevant[:k]
    #             num_relevant = np.sum(relevant)
                
    #             # Precision@K
    #             precision = np.sum(relevant_at_k) / k
    #             metrics['precision_at_k'][k].append(precision)
                
    #             # Recall@K（假设相关文档总数就是检索到的相关数）
    #             if num_relevant > 0:
    #                 recall = np.sum(relevant_at_k) / num_relevant
    #                 metrics['recall_at_k'][k].append(recall)
            
    #         # MRR
    #         first_relevant_rank = np.where(relevant)[0]
    #         if len(first_relevant_rank) > 0:
    #             metrics['mrr'].append(1.0 / (first_relevant_rank[0] + 1))
    #         else:
    #             metrics['mrr'].append(0.0)
        
    #     # 计算平均值
    #     result = {
    #         'precision_at_k': {k: np.mean(v) for k, v in metrics['precision_at_k'].items()},
    #         'recall_at_k': {k: np.mean(v) for k, v in metrics['recall_at_k'].items()},
    #         'mrr': np.mean(metrics['mrr']),
    #         'avg_comment_similarity': np.mean(metrics['avg_similarity'])
    #     }
        
    #     return result


def run_evaluation(
    original_data_file: str,
    test_data_file: str,
    checkpoint_path: str,
    output_code_file: str,
    output_comment_file: str,
    top_k: int = 30,
):
    """
    运行评估流程
    """
    print("="*70)
    print("检索评估")
    print("="*70)
    
    # 检查文件
    if not os.path.exists(original_data_file):
        print(f"错误: 找不到原始数据: {original_data_file}")
        return
    
    if not os.path.exists(test_data_file):
        print(f"错误: 找不到测试数据: {test_data_file}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型: {checkpoint_path}")
        return
    
    # 加载模型
    print("\n加载模型...")
    from transformers import AutoTokenizer
    from unixcoder_contrastive_trainer import UniXcoderContrastiveModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
    model = UniXcoderContrastiveModel(
        model_name="microsoft/unixcoder-base-nine",
        projection_dim=256
    )
    
    # 加载权重
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ 模型加载完成")
    print(f"    Epoch: {checkpoint['epoch']}")
    print(f"    Val Loss: {checkpoint['val_loss']:.4f}")
    
    # 初始化评估器
    evaluator = RetrievalEvaluator(model, tokenizer, device)
    
    # 检索并保存
    metrics = evaluator.retrieve_and_save(
        original_data_file=original_data_file,
        test_data_file=test_data_file,
        output_code_file=output_code_file,
        output_comment_file=output_comment_file,
        top_k=top_k
    )

#     # 打印结果
#     print("\n" + "="*70)
#     print("评估结果")
#     print("="*70)
    
#     print("\n1. Precision@K (相关候选的比例):")
#     for k, precision in sorted(metrics['precision_at_k'].items()):
#         print(f"   Precision@{k:2d}: {precision:.4f}")
    
#     print("\n2. Recall@K (找到的相关候选比例):")
#     for k, recall in sorted(metrics['recall_at_k'].items()):
#         print(f"   Recall@{k:2d}: {recall:.4f}")
    
#     print(f"\n3. MRR (平均倒数排名): {metrics['mrr']:.4f}")
    
#     print(f"\n4. 平均 Comment 相似度: {metrics['avg_comment_similarity']:.4f}")
    
#     print("\n" + "="*70)
#     print("说明")
#     print("="*70)
#     print(f"""
# - Comment 相似度阈值: 0.7 (超过此值认为相关)
# - 评估基于: comment 语义相似性
# - 输出文件:
#   * retrieval_top30_code.txt - 每30行对应一个测试样本的候选 diffs
#   * retrieval_top30_comment.txt - 每30行对应一个测试样本的候选 comments
# - 格式: 
#   * 第 0-29 行: test_data[0] 的候选
#   * 第 30-59 行: test_data[1] 的候选
#   * ...
#     """)
    
#     # 保存结果
#     results_file = 'evaluation_results.json'
#     with open(results_file, 'w', encoding='utf-8') as f:
#         json.dump(metrics, f, indent=2)
#     print(f"\n✓ 结果保存到: {results_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        top_k = int(sys.argv[1])
    else:
        top_k = 30
    
    print(f"Top-K: {top_k}")

    ## 实现不同检索数据库大小的评估，由于msg的训练集没有lang属性所以我们用ref的训练集合作为全局检索库
    for rate in [2, 5, 10]:
        run_evaluation(
            original_data_file=f"D:/code_review_comments_gen/my_finetune_project/dataset/CodeReviewer_dataset/Code_Refinement/ref-train-{rate}%.jsonl",
            test_data_file="D:/code_review_comments_gen/my_finetune_project/dataset/CodeReviewer_dataset/Comment_Generation/msg-test.jsonl",
            checkpoint_path='./checkpoints/best_model.pt',
            output_code_file=f'./ref_rag_candidates_{rate}%/test_to_train_retrieval_top30_code.txt',
            output_comment_file=f'./ref_rag_candidates_{rate}%/test_to_train_retrieval_top30_comment.txt',
            top_k=top_k
        )
    
    # for rate in [20, 40, 60, 80, 100]:
    #     run_evaluation(
    #         original_data_file=f"D:/code_review_comments_gen/my_finetune_project/dataset/CodeReviewer_dataset/Code_Refinement/ref-train-{rate}%.jsonl",
    #         test_data_file="D:/code_review_comments_gen/my_finetune_project/dataset/CodeReviewer_dataset/Comment_Generation/msg-test.jsonl",
    #         checkpoint_path='./checkpoints/best_model.pt',
    #         output_code_file=f'./ref_rag_candidates_{rate}%/test_to_train_retrieval_top30_code.txt',
    #         output_comment_file=f'./ref_rag_candidates_{rate}%/test_to_train_retrieval_top30_comment.txt',
    #         top_k=top_k
    #     )

    ## 以下是原始全部检索数据库的检索
    # run_evaluation(
    #     original_data_file="/root/autodl-tmp/pure_finetune_project/dataset/CodeReviewer_dataset/Comment_Generation/msg-train-100%.jsonl",
    #     test_data_file="/root/autodl-tmp/pure_finetune_project/dataset/CodeReviewer_dataset/Comment_Generation/msg-test.jsonl",
    #     checkpoint_path='./checkpoints/best_model.pt',
    #     output_code_file='./msg_rag_candidates/test_to_train_retrieval_top30_code.txt',
    #     output_comment_file='./msg_rag_candidates/test_to_train_retrieval_top30_comment.txt',
    #     top_k=top_k
    # )
    # run_evaluation(
    #     original_data_file="/root/autodl-tmp/pure_finetune_project/dataset/CodeReviewer_dataset/Comment_Generation/msg-train.jsonl",
    #     test_data_file="/root/autodl-tmp/pure_finetune_project/dataset/CodeReviewer_dataset/Comment_Generation/msg-train.jsonl",
    #     checkpoint_path='./checkpoints/best_model.pt',
    #     output_code_file='./msg_rag_candidates/train_to_train_retrieval_top30_code.txt',
    #     output_comment_file='./msg_rag_candidates/train_to_train_retrieval_top30_comment.txt',
    #     top_k=top_k
    # )
    # run_evaluation(
    #     original_data_file="/root/autodl-tmp/pure_finetune_project/dataset/CodeReviewer_dataset/Comment_Generation/msg-train.jsonl",
    #     test_data_file="/root/autodl-tmp/pure_finetune_project/dataset/CodeReviewer_dataset/Comment_Generation/msg-valid.jsonl",
    #     checkpoint_path='./checkpoints/best_model.pt',
    #     output_code_file='./msg_rag_candidates/val_to_train_retrieval_top30_code.txt',
    #     output_comment_file='./msg_rag_candidates/val_to_train_retrieval_top30_comment.txt',
    #     top_k=top_k
    # )