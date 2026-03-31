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
    """Build training data for contrastive learning."""
    
    def __init__(self, comment_model_name='all-MiniLM-L6-v2'):
        self.comment_encoder = SentenceTransformer(comment_model_name)
        self.data = []
        self.comment_embeddings = None
        
    def load_data(self, file_path: str):
        """Load raw data."""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        print(f"Loaded {len(self.data)} records")
        
    def extract_change_features(self, diff: str) -> Dict:
        """Extract change features from a diff."""
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
        
        # Detect common code patterns
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
        """Compute similarity between change features."""
        # Size similarity (use log scale to reduce large-value impact)
        size_sim = 1 - abs(
            np.log1p(feat1['total_change']) - np.log1p(feat2['total_change'])
        ) / max(np.log1p(feat1['total_change']), np.log1p(feat2['total_change']), 1)
        
        # Type similarity
        type_sim = (
            (feat1['has_addition'] == feat2['has_addition']) +
            (feat1['has_deletion'] == feat2['has_deletion']) +
            (feat1['has_modification'] == feat2['has_modification'])
        ) / 3
        
        # Pattern similarity
        pattern_keys = [k for k in feat1.keys() if k.startswith('pattern_')]
        pattern_sim = sum(feat1[k] == feat2[k] for k in pattern_keys) / max(len(pattern_keys), 1)
        
        # Weighted combination
        return 0.3 * size_sim + 0.3 * type_sim + 0.4 * pattern_sim
    
    def build_triplets(self, 
                       comment_threshold: float = 0.7,
                       change_threshold: float = 0.5,
                       negative_samples: int = 5) -> List[Dict]:
        """Build triplets (anchor, positive, negative)."""
        
        print("Extracting change features...")
        for item in self.data:
            item['change_features'] = self.extract_change_features(item['hunk'])
        
        print("Encoding comment semantics...")
        comments = [item['comment'] for item in self.data]
        self.comment_embeddings = self.comment_encoder.encode(comments, show_progress_bar=True)
        
        print("Computing comment similarity matrix...")
        # comment_sim_matrix = cosine_similarity(self.comment_embeddings)
        norms = np.linalg.norm(self.comment_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = self.comment_embeddings / norms
        comment_sim_matrix = np.dot(normalized, normalized.T)
        # Group indices by language
        lang_groups = defaultdict(list)
        for idx, item in enumerate(self.data):
            lang_groups[item['lang']].append(idx)
        
        triplets = []
        
        print("Building triplets...")
        for anchor_idx in range(len(self.data)):
            anchor = self.data[anchor_idx]
            anchor_lang = anchor['lang']
            
            # Find positive/negative samples within the same language
            same_lang_indices = lang_groups[anchor_lang]
            
            # Find positive samples: similar comments + similar change features
            positive_candidates = []
            for idx in same_lang_indices:
                if idx == anchor_idx:
                    continue
                    
                comment_sim = comment_sim_matrix[anchor_idx][idx]
                change_sim = self.compute_change_similarity(
                    anchor['change_features'],
                    self.data[idx]['change_features']
                )
                
                # Combined similarity
                combined_sim = 0.7 * comment_sim + 0.3 * change_sim
                
                if comment_sim >= comment_threshold and change_sim >= change_threshold:
                    positive_candidates.append((idx, combined_sim))
            
            if not positive_candidates:
                continue
            
            # Choose the most similar one as the positive sample
            positive_candidates.sort(key=lambda x: x[1], reverse=True)
            positive_idx = positive_candidates[0][0]
            
            # Find negative samples: dissimilar comments or dissimilar change features
            negative_candidates = []
            for idx in same_lang_indices:
                if idx == anchor_idx or idx == positive_idx:
                    continue
                    
                comment_sim = comment_sim_matrix[anchor_idx][idx]
                change_sim = self.compute_change_similarity(
                    anchor['change_features'],
                    self.data[idx]['change_features']
                )
                
                # A negative sample should be dissimilar in at least one dimension
                if comment_sim < 0.5 or change_sim < 0.3:
                    dissim_score = 1 - (0.7 * comment_sim + 0.3 * change_sim)
                    negative_candidates.append((idx, dissim_score))
            
            # Sample multiple negative samples
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
        
        print(f"Built {len(triplets)} triplets")
        return triplets
    
    def build_triplets_fast(self,
                           comment_threshold: float = 0.7,
                           change_threshold: float = 0.5,
                           negative_samples: int = 5,
                           top_k_candidates: int = 100) -> List[Dict]:
        """
        Build triplets quickly (accelerated with FAISS).
        
        Args:
            comment_threshold: Comment similarity threshold.
            change_threshold: Change-feature similarity threshold.
            negative_samples: Number of negative samples.
            top_k_candidates: Number of candidates retrieved from FAISS (larger is more accurate but slower).
        
        Returns:
            A list of triplets.
        """
        print("\n" + "="*70)
        print("Fast triplet construction (FAISS-accelerated)")
        print("="*70)
        
        # Step 1: Extract change features
        print("\n[1/6] Extracting change features...")
        for item in tqdm(self.data, desc="Extracting features"):
            item['change_features'] = self.extract_change_features(item['hunk'])
        
        # Step 2: Encode comments (in batches)
        print("\n[2/6] Encoding comments...")
        comments = [item['comment'] for item in self.data]
        
        batch_size = 256
        embeddings_list = []
        for i in tqdm(range(0, len(comments), batch_size), desc="Batch encoding"):
            batch = comments[i:i+batch_size]
            batch_emb = self.comment_encoder.encode(
                batch, 
                show_progress_bar=False,
                batch_size=32
            )
            embeddings_list.append(batch_emb)
        
        self.comment_embeddings = np.vstack(embeddings_list).astype('float32')
        print(f"   Embeddings shape: {self.comment_embeddings.shape}")
        
        # Step 3: Group indices by language
        print("\n[3/6] Building language-group index...")
        lang_groups = defaultdict(list)
        for idx, item in enumerate(self.data):
            lang_groups[item['lang']].append(idx)
        
        print(f"   Number of languages: {len(lang_groups)}")
        for lang, indices in sorted(lang_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            print(f"   - {lang}: {len(indices)} items")
        
        # Step 4: Build a FAISS index for each language
        print("\n[4/6] Building FAISS index for each language...")
        lang_indices = {}
        
        for lang, indices in tqdm(lang_groups.items(), desc="Building indexes"):
            if len(indices) < 10:  # Skip languages with too few samples
                continue
            
            # Extract embeddings for this language
            lang_embs = self.comment_embeddings[indices]
            
            # Normalize (for cosine similarity)
            norms = np.linalg.norm(lang_embs, axis=1, keepdims=True)
            lang_embs_normalized = lang_embs / (norms + 1e-8)
            
            # Build FAISS index
            dimension = lang_embs_normalized.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner-product index (cosine similarity)
            index.add(lang_embs_normalized)
            
            lang_indices[lang] = {
                'index': index,
                'data_indices': indices,
                'embeddings': lang_embs_normalized
            }
        
        print(f"   Successfully built {len(lang_indices)} language indexes")
        
        # Step 5: Build triplets quickly
        print("\n[5/6] Building triplets (using FAISS approximate search)...")
        triplets = []
        
        for anchor_idx in tqdm(range(len(self.data)), desc="Processing samples"):
            anchor = self.data[anchor_idx]
            anchor_lang = anchor['lang']
            
            # Skip if this language has too few samples
            if anchor_lang not in lang_indices:
                continue
            
            lang_info = lang_indices[anchor_lang]
            
            # Find anchor position in this language index
            try:
                anchor_pos = lang_info['data_indices'].index(anchor_idx)
            except ValueError:
                continue
            
            # Get anchor embedding
            anchor_emb = lang_info['embeddings'][anchor_pos:anchor_pos+1]
            
            # Use FAISS to quickly retrieve top-k similar samples
            k = min(top_k_candidates, len(lang_info['data_indices']))
            scores, indices = lang_info['index'].search(anchor_emb, k)
            
            # Process retrieval results
            positive_candidates = []
            negative_candidates = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == anchor_pos:  # Skip self
                    continue
                
                # Get actual data index
                candidate_idx = lang_info['data_indices'][idx]
                candidate = self.data[candidate_idx]
                
                # Compute change-feature similarity
                change_sim = self.compute_change_similarity(
                    anchor['change_features'],
                    candidate['change_features']
                )
                
                # Comment similarity is the score returned by FAISS
                comment_sim = float(score)
                
                # Combined similarity
                combined_sim = 0.7 * comment_sim + 0.3 * change_sim
                
                # Decide whether it is a positive or negative sample
                if comment_sim >= comment_threshold and change_sim >= change_threshold:
                    positive_candidates.append((candidate_idx, combined_sim))
                elif comment_sim < 0.5 or change_sim < 0.3:
                    dissim_score = 1 - combined_sim
                    negative_candidates.append((candidate_idx, dissim_score))
            
            # Must have enough positive and negative samples
            if not positive_candidates or len(negative_candidates) < negative_samples:
                continue
            
            # Select the most similar positive sample
            positive_candidates.sort(key=lambda x: x[1], reverse=True)
            positive_idx = positive_candidates[0][0]
            
            # Select the most dissimilar negative samples
            negative_candidates.sort(key=lambda x: x[1], reverse=True)
            selected_negatives = [idx for idx, _ in negative_candidates[:negative_samples]]
            
            # Build triplet
            triplets.append({
                'anchor': anchor_idx,
                'positive': positive_idx,
                'negatives': selected_negatives,
                'anchor_data': anchor,
                'positive_data': self.data[positive_idx],
                'negative_data': [self.data[i] for i in selected_negatives]
            })
        
        print(f"\n   Built {len(triplets)} triplets")
        print(f"   Triplet ratio: {len(triplets) / len(self.data) * 100:.1f}%")
        
        # Step 6: Summary statistics
        print("\n[6/6] Statistics:")
        if triplets:
            # Count triplets per language
            lang_triplet_count = defaultdict(int)
            for t in triplets:
                lang_triplet_count[t['anchor_data']['lang']] += 1
            
            print(f"   Total triplets: {len(triplets)}")
            print(f"   Covered languages: {len(lang_triplet_count)}")
            print(f"   Top-5 languages:")
            for lang, count in sorted(lang_triplet_count.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                print(f"   - {lang}: {count} triplets")
        
        return triplets

    def save_triplets(self, triplets: List[Dict], output_path: str):
        """Save the constructed triplet data."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for triplet in triplets:
                # Save only indices and necessary fields
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
        print(f"Saved to {output_path}")


# Usage example
if __name__ == "__main__":
    builder = CodeReviewDataBuilder()
    
    # Load data
    builder.load_data("../your_full_database.jsonl")
    
    # Build triplets
    triplets = builder.build_triplets_fast(
        comment_threshold=0.7,
        change_threshold=0.5,
        negative_samples=5,
        top_k_candidates=100
    )
    
    # Save
    builder.save_triplets(triplets, "../triplets.jsonl")