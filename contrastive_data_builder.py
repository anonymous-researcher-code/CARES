import json
import re
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
from tqdm import tqdm
import faiss
import multiprocessing as mp
from functools import partial

class CodeReviewDataBuilder:
    """构建对比学习训练数据"""
    
    def __init__(self, comment_model_name='all-MiniLM-L6-v2'):
        self.comment_encoder = SentenceTransformer(comment_model_name)
        self.data = []
        self.comment_embeddings = None
        
    def load_data(self, file_path: str):
        """加载原始数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        print(f"加载了 {len(self.data)} 条数据")
        
    def extract_change_features(self, diff: str) -> Dict:
        """提取diff的变更特征"""
        lines = diff.split('\n')
        
        added_lines = [l for l in lines if l.startswith('+') and not l.startswith('+++')]
        deleted_lines = [l for l in lines if l.startswith('-') and not l.startswith('---')]
        
        features = {
            'added_count': len(added_lines),
            'deleted_count': len(deleted_lines),
            'total_change': len(added_lines) + len(deleted_lines),
            'change_ratio': len(added_lines) / max(len(deleted_lines), 1),
            'has_addition': len(added_lines) > 0,
            'has_deletion': len(deleted_lines) > 0,
            'has_modification': len(added_lines) > 0 and len(deleted_lines) > 0
        }
        
        # 检测常见的代码模式
        patterns = {
            'null_check': r'(if.*null|null.*check|nullptr)',
            'error_handling': r'(try|catch|exception|error)',
            'resource_mgmt': r'(close|dispose|release|free)',
            'validation': r'(validate|check|verify|assert)',
            'naming': r'(rename|naming|name)',
            'performance': r'(optimize|performance|efficient)',
            'security': r'(security|vulnerability|sanitize)',
        }
        
        diff_text = diff.lower()
        for pattern_name, pattern in patterns.items():
            features[f'pattern_{pattern_name}'] = bool(re.search(pattern, diff_text))
            
        return features
    
    def compute_change_similarity(self, feat1: Dict, feat2: Dict) -> float:
        """计算变更特征的相似度"""
        # 规模相似度（使用对数避免大数值影响）
        size_sim = 1 - abs(
            np.log1p(feat1['total_change']) - np.log1p(feat2['total_change'])
        ) / max(np.log1p(feat1['total_change']), np.log1p(feat2['total_change']), 1)
        
        # 类型相似度
        type_sim = (
            (feat1['has_addition'] == feat2['has_addition']) +
            (feat1['has_deletion'] == feat2['has_deletion']) +
            (feat1['has_modification'] == feat2['has_modification'])
        ) / 3
        
        # 模式相似度
        pattern_keys = [k for k in feat1.keys() if k.startswith('pattern_')]
        pattern_sim = sum(feat1[k] == feat2[k] for k in pattern_keys) / max(len(pattern_keys), 1)
        
        # 加权组合
        return 0.3 * size_sim + 0.3 * type_sim + 0.4 * pattern_sim
    
    def build_triplets(self, 
                       comment_threshold: float = 0.7,
                       change_threshold: float = 0.5,
                       negative_samples: int = 5) -> List[Dict]:
        """构建三元组 (anchor, positive, negative)"""
        
        print("提取变更特征...")
        for item in self.data:
            item['change_features'] = self.extract_change_features(item['hunk'])
        
        print("编码评论语义...")
        comments = [item['comment'] for item in self.data]
        self.comment_embeddings = self.comment_encoder.encode(comments, show_progress_bar=True)
        
        print("计算评论相似度矩阵...")
        # comment_sim_matrix = cosine_similarity(self.comment_embeddings)
        norms = np.linalg.norm(self.comment_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = self.comment_embeddings / norms
        comment_sim_matrix = np.dot(normalized, normalized.T)
        # 按语言分组索引
        lang_groups = defaultdict(list)
        for idx, item in enumerate(self.data):
            lang_groups[item['lang']].append(idx)
        
        triplets = []
        
        print("构建三元组...")
        for anchor_idx in range(len(self.data)):
            anchor = self.data[anchor_idx]
            anchor_lang = anchor['lang']
            
            # 在相同语言内寻找正负样本
            same_lang_indices = lang_groups[anchor_lang]
            
            # 找正样本：comment相似 + 变更特征相似
            positive_candidates = []
            for idx in same_lang_indices:
                if idx == anchor_idx:
                    continue
                    
                comment_sim = comment_sim_matrix[anchor_idx][idx]
                change_sim = self.compute_change_similarity(
                    anchor['change_features'],
                    self.data[idx]['change_features']
                )
                
                # 综合相似度
                combined_sim = 0.7 * comment_sim + 0.3 * change_sim
                
                if comment_sim >= comment_threshold and change_sim >= change_threshold:
                    positive_candidates.append((idx, combined_sim))
            
            if not positive_candidates:
                continue
            
            # 选择最相似的作为正样本
            positive_candidates.sort(key=lambda x: x[1], reverse=True)
            positive_idx = positive_candidates[0][0]
            
            # 找负样本：comment不相似或变更特征不相似
            negative_candidates = []
            for idx in same_lang_indices:
                if idx == anchor_idx or idx == positive_idx:
                    continue
                    
                comment_sim = comment_sim_matrix[anchor_idx][idx]
                change_sim = self.compute_change_similarity(
                    anchor['change_features'],
                    self.data[idx]['change_features']
                )
                
                # 负样本应该在至少一个维度上不相似
                if comment_sim < 0.5 or change_sim < 0.3:
                    dissim_score = 1 - (0.7 * comment_sim + 0.3 * change_sim)
                    negative_candidates.append((idx, dissim_score))
            
            # 采样多个负样本
            if len(negative_candidates) < negative_samples:
                continue
                
            negative_candidates.sort(key=lambda x: x[1], reverse=True)
            selected_negatives = [idx for idx, _ in negative_candidates[:negative_samples]]
            
            triplets.append({
                'anchor': anchor_idx,
                'positive': positive_idx,
                'negatives': selected_negatives,
                'anchor_data': anchor,
                'positive_data': self.data[positive_idx],
                'negative_data': [self.data[i] for i in selected_negatives]
            })
        
        print(f"构建了 {len(triplets)} 个三元组")
        return triplets
    
    def build_triplets_fast(self,
                           comment_threshold: float = 0.7,
                           change_threshold: float = 0.5,
                           negative_samples: int = 5,
                           top_k_candidates: int = 100) -> List[Dict]:
        """
        快速构建三元组（使用FAISS加速）
        
        Args:
            comment_threshold: comment相似度阈值
            change_threshold: 变更特征相似度阈值
            negative_samples: 负样本数量
            top_k_candidates: 从FAISS检索的候选数量（越大越准确但越慢）
        
        Returns:
            三元组列表
        """
        print("\n" + "="*70)
        print("快速三元组构建（使用FAISS加速）")
        print("="*70)
        
        # Step 1: 提取变更特征
        print("\n[1/6] 提取变更特征...")
        for item in tqdm(self.data, desc="提取特征"):
            item['change_features'] = self.extract_change_features(item['hunk'])
        
        # Step 2: 编码评论（分批）
        print("\n[2/6] 编码评论...")
        comments = [item['comment'] for item in self.data]
        
        batch_size = 256
        embeddings_list = []
        for i in tqdm(range(0, len(comments), batch_size), desc="批量编码"):
            batch = comments[i:i+batch_size]
            batch_emb = self.comment_encoder.encode(
                batch, 
                show_progress_bar=False,
                batch_size=32
            )
            embeddings_list.append(batch_emb)
        
        self.comment_embeddings = np.vstack(embeddings_list).astype('float32')
        print(f"   Embeddings shape: {self.comment_embeddings.shape}")
        
        # Step 3: 按语言分组索引
        print("\n[3/6] 构建语言分组索引...")
        lang_groups = defaultdict(list)
        for idx, item in enumerate(self.data):
            lang_groups[item['lang']].append(idx)
        
        print(f"   语言数量: {len(lang_groups)}")
        for lang, indices in sorted(lang_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            print(f"   - {lang}: {len(indices)} 条")
        
        # Step 4: 为每种语言构建FAISS索引
        print("\n[4/6] 为每种语言构建FAISS索引...")
        lang_indices = {}
        
        for lang, indices in tqdm(lang_groups.items(), desc="构建索引"):
            if len(indices) < 10:  # 太少的语言跳过
                continue
            
            # 提取该语言的embeddings
            lang_embs = self.comment_embeddings[indices]
            
            # 归一化（用于余弦相似度）
            norms = np.linalg.norm(lang_embs, axis=1, keepdims=True)
            lang_embs_normalized = lang_embs / (norms + 1e-8)
            
            # 构建FAISS索引
            dimension = lang_embs_normalized.shape[1]
            index = faiss.IndexFlatIP(dimension)  # 内积索引（余弦相似度）
            index.add(lang_embs_normalized)
            
            lang_indices[lang] = {
                'index': index,
                'data_indices': indices,
                'embeddings': lang_embs_normalized
            }
        
        print(f"   成功构建 {len(lang_indices)} 个语言索引")
        
        # Step 5: 快速构建三元组
        print("\n[5/6] 构建三元组（使用FAISS近似搜索）...")
        triplets = []
        
        for anchor_idx in tqdm(range(len(self.data)), desc="处理样本"):
            anchor = self.data[anchor_idx]
            anchor_lang = anchor['lang']
            
            # 如果该语言样本太少，跳过
            if anchor_lang not in lang_indices:
                continue
            
            lang_info = lang_indices[anchor_lang]
            
            # 在本语言的索引中找到anchor的位置
            try:
                anchor_pos = lang_info['data_indices'].index(anchor_idx)
            except ValueError:
                continue
            
            # 获取anchor的embedding
            anchor_emb = lang_info['embeddings'][anchor_pos:anchor_pos+1]
            
            # 使用FAISS快速检索top-k相似样本
            k = min(top_k_candidates, len(lang_info['data_indices']))
            scores, indices = lang_info['index'].search(anchor_emb, k)
            
            # 处理检索结果
            positive_candidates = []
            negative_candidates = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == anchor_pos:  # 跳过自己
                    continue
                
                # 获取实际的数据索引
                candidate_idx = lang_info['data_indices'][idx]
                candidate = self.data[candidate_idx]
                
                # 计算变更特征相似度
                change_sim = self.compute_change_similarity(
                    anchor['change_features'],
                    candidate['change_features']
                )
                
                # comment相似度就是FAISS返回的score
                comment_sim = float(score)
                
                # 综合相似度
                combined_sim = 0.7 * comment_sim + 0.3 * change_sim
                
                # 判断是正样本还是负样本
                if comment_sim >= comment_threshold and change_sim >= change_threshold:
                    positive_candidates.append((candidate_idx, combined_sim))
                elif comment_sim < 0.5 or change_sim < 0.3:
                    dissim_score = 1 - combined_sim
                    negative_candidates.append((candidate_idx, dissim_score))
            
            # 必须有足够的正负样本
            if not positive_candidates or len(negative_candidates) < negative_samples:
                continue
            
            # 选择最相似的正样本
            positive_candidates.sort(key=lambda x: x[1], reverse=True)
            positive_idx = positive_candidates[0][0]
            
            # 选择最不相似的负样本
            negative_candidates.sort(key=lambda x: x[1], reverse=True)
            selected_negatives = [idx for idx, _ in negative_candidates[:negative_samples]]
            
            # 构建三元组
            triplets.append({
                'anchor': anchor_idx,
                'positive': positive_idx,
                'negatives': selected_negatives,
                'anchor_data': anchor,
                'positive_data': self.data[positive_idx],
                'negative_data': [self.data[i] for i in selected_negatives]
            })
        
        print(f"\n   构建了 {len(triplets)} 个三元组")
        print(f"   三元组比例: {len(triplets) / len(self.data) * 100:.1f}%")
        
        # Step 6: 保存统计信息
        print("\n[6/6] 统计信息:")
        if triplets:
            # 统计每种语言的三元组数量
            lang_triplet_count = defaultdict(int)
            for t in triplets:
                lang_triplet_count[t['anchor_data']['lang']] += 1
            
            print(f"   总三元组: {len(triplets)}")
            print(f"   覆盖的语言: {len(lang_triplet_count)}")
            print(f"   Top-5 语言:")
            for lang, count in sorted(lang_triplet_count.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                print(f"   - {lang}: {count} 个三元组")
        
        return triplets

    def save_triplets(self, triplets: List[Dict], output_path: str):
        """保存构建的三元组数据"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for triplet in triplets:
                # 只保存索引和必要信息
                simplified = {
                    'anchor_idx': triplet['anchor'],
                    'positive_idx': triplet['positive'],
                    'negative_indices': triplet['negatives'],
                    'anchor_diff': triplet['anchor_data']['hunk'],
                    'anchor_comment': triplet['anchor_data']['comment'],
                    'positive_diff': triplet['positive_data']['hunk'],
                    'positive_comment': triplet['positive_data']['comment'],
                    'negative_diffs': [d['hunk'] for d in triplet['negative_data']],
                    'negative_comments': [d['comment'] for d in triplet['negative_data']],
                    'lang': triplet['anchor_data']['lang']
                }
                f.write(json.dumps(simplified, ensure_ascii=False) + '\n')
        print(f"保存到 {output_path}")


# 使用示例
if __name__ == "__main__":
    builder = CodeReviewDataBuilder()
    
    # 加载数据
    builder.load_data("D:/code_review_comments_gen/my_finetune_project/dataset/CodeReviewer_dataset/Code_Refinement/ref-train.jsonl")
    
    # 构建三元组
    triplets = builder.build_triplets_fast(
        comment_threshold=0.7,
        change_threshold=0.5,
        negative_samples=5,
        top_k_candidates=100
    )
    
    # 保存
    builder.save_triplets(triplets, "D:/code_review_comments_gen/my_finetune_project/RAG-codereviewer-finetune/code/contrastive_finetune_unixcoder/triplets_train.jsonl")