"""
完整的代码审查检索系统流程
包含数据处理、模型训练、评估测试的全部步骤
"""

import os
import json
import torch
from sklearn.model_selection import train_test_split
from contrastive_data_builder import CodeReviewDataBuilder
from transformers import AutoTokenizer
from unixcoder_contrastive_trainer import (
    UniXcoderContrastiveModel,
    ContrastiveDataset,
    ContrastiveTrainer
)
from torch.utils.data import DataLoader

def split_dataset(input_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """划分数据集"""
    print("Step 1: 划分数据集...")
    
    # 加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 划分
    train_val, test = train_test_split(data, test_size=1-train_ratio-val_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)
    
    # 保存
    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        output_file = f'data_{split_name}.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  {split_name}: {len(split_data)} samples -> {output_file}")
    
    return len(train), len(val), len(test)


def build_triplets_for_all_splits():
    """为所有数据集构建三元组"""
    print("\nStep 2: 构建三元组...")
    
    for split in ['train', 'val', 'test']:
        triplets_file = f'triplets_{split}.jsonl'
        if os.path.exists(triplets_file):
            print(f"\n{triplets_file} 已存在，跳过构建三元组，直接使用。")
            continue
        print(f"\n处理 {split} 集...")
        builder = CodeReviewDataBuilder()
        builder.load_data(f'data_{split}.jsonl')
        
        triplets = builder.build_triplets_fast(
            comment_threshold=0.7,
            change_threshold=0.5,
            negative_samples=5,
            top_k_candidates=100
        )
        
        builder.save_triplets(triplets, f'triplets_{split}.jsonl')


def train_model():
    """训练模型"""
    print("\nStep 3: 训练模型...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
    
    # 准备数据
    train_dataset = ContrastiveDataset('triplets_train.jsonl', tokenizer)
    val_dataset = ContrastiveDataset('triplets_val.jsonl', tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # 初始化模型
    model = UniXcoderContrastiveModel(
        model_name="microsoft/unixcoder-base-nine",
        projection_dim=256
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 训练
    os.makedirs('checkpoints', exist_ok=True)
    trainer = ContrastiveTrainer(
        model, 
        train_loader, 
        val_loader, 
        device=device,
        lr=2e-5
    )
    
    trainer.train(num_epochs=10, save_dir='./checkpoints')


# 这个也建议删除
def inference_demo():
    """推理演示"""
    print("\nStep 5: 推理演示...")
    
    from transformers import AutoTokenizer
    from unixcoder_contrastive_trainer import UniXcoderContrastiveModel
    # from retrieval_evaluation import CodeReviewRetriever这个文件已经被更新了
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
    model = UniXcoderContrastiveModel(
        model_name="microsoft/unixcoder-base-nine",
        projection_dim=256
    )
    
    checkpoint = torch.load('./checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 初始化检索器
    retriever = CodeReviewRetriever(model, tokenizer, device)
    retriever.load_index('code_review.index', 'indexed_data.jsonl')
    
    # 示例查询
    query = {
        'hunk': '''
@@ -10,7 +10,10 @@ def process_data(data):
-    result = data.get('value')
+    if data is None:
+        return None
+    result = data.get('value')
     return result
''',
        'comment': 'Added null check to prevent NullPointerException',
        'lang': 'python'
    }
    
    print("\n查询:")
    print(f"Diff:\n{query['hunk']}")
    print(f"Comment: {query['comment']}")
    print(f"Language: {query['lang']}")
    
    results = retriever.retrieve(query, top_k=5)
    
    print("\n检索结果:")
    for rank, (result, score) in enumerate(results, 1):
        print(f"\n{'='*50}")
        print(f"Rank {rank} (相似度: {score:.4f})")
        print(f"Comment: {result['comment']}")
        print(f"Language: {result['lang']}")
        print(f"Diff:\n{result['hunk'][:300]}...")


def main():
    """主流程"""
    print("="*70)
    print("代码审查检索系统 - 完整流程")
    print("="*70)
    
    # 检查原始数据
    if not os.path.exists("/root/autodl-tmp/pure_finetune_project/dataset/CodeReviewer_dataset/Code_Refinement/ref-train.jsonl"):
        print("错误: 找不到 code_review_data.jsonl")
        print("请准备您的原始数据文件，格式为每行一个JSON对象，包含 'hunk', 'comment', 'lang' 字段")
        return
    
    # Step 1: 划分数据集
    # 如果已存在 data_train.jsonl, data_val.jsonl, data_test.jsonl，则跳过划分
    data_files = ["data_train.jsonl", "data_val.jsonl", "data_test.jsonl"]
    if all(os.path.exists(f) for f in data_files):
        print("检测到已存在数据集文件，跳过数据集划分。")
        train_size = sum(1 for _ in open("data_train.jsonl"))
        val_size = sum(1 for _ in open("data_val.jsonl"))
        test_size = sum(1 for _ in open("data_test.jsonl"))
    else:
        train_size, val_size, test_size = split_dataset("/root/autodl-tmp/pure_finetune_project/dataset/CodeReviewer_dataset/Code_Refinement/ref-train.jsonl")
    
    print(f"\n总计: {train_size + val_size + test_size} samples")
    
    # Step 2: 构建三元组
    build_triplets_for_all_splits()
    
    # Step 3: 训练模型
    train_model()
    
    # Step 4: 评估模型
    # evaluate_model()
    
    # Step 5: 推理演示
    # inference_demo()
    
    print("\n" + "="*70)
    print("流程完成!")
    print("="*70)
    print("\n生成的文件:")
    print("  - data_train.jsonl, data_val.jsonl, data_test.jsonl")
    print("  - triplets_train.jsonl, triplets_val.jsonl, triplets_test.jsonl")
    print("  - checkpoints/best_model.pt")
    print("  - code_review.index, indexed_data.jsonl")
    print("\n使用方法:")
    print("  1. 加载模型: model.load_state_dict(torch.load('checkpoints/best_model.pt'))")
    print("  2. 初始化检索器: retriever = CodeReviewRetriever(model, tokenizer, device)")
    print("  3. 加载索引: retriever.load_index('code_review.index', 'indexed_data.jsonl')")
    print("  4. 检索: results = retriever.retrieve(query, top_k=10)")


if __name__ == "__main__":
    main()