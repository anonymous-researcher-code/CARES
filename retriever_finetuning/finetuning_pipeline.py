"""
Complete code review retrieval system pipeline.
Includes all steps for data processing, model training, and evaluation/testing.
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
    """Split the dataset."""
    print("Step 1: Split the dataset...")
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Split
    train_val, test = train_test_split(data, test_size=1-train_ratio-val_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)
    
    # Save
    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        output_file = f'data_{split_name}.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  {split_name}: {len(split_data)} samples -> {output_file}")
    
    return len(train), len(val), len(test)


def build_triplets_for_all_splits():
    """Build triplets for all dataset splits."""
    print("\nStep 2: Build triplets...")
    
    for split in ['train', 'val', 'test']:
        triplets_file = f'triplets_{split}.jsonl'
        if os.path.exists(triplets_file):
            print(f"\n{triplets_file} already exists, skip triplet construction and use it directly.")
            continue
        print(f"\nProcessing {split} split...")
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
    """Train the model."""
    print("\nStep 3: Train the model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
    
    # Prepare data
    train_dataset = ContrastiveDataset('triplets_train.jsonl', tokenizer)
    val_dataset = ContrastiveDataset('triplets_val.jsonl', tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Initialize model
    model = UniXcoderContrastiveModel(
        model_name="microsoft/unixcoder-base-nine",
        projection_dim=256
    )
    
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Train
    os.makedirs('checkpoints', exist_ok=True)
    trainer = ContrastiveTrainer(
        model, 
        train_loader, 
        val_loader, 
        device=device,
        lr=2e-5
    )
    
    trainer.train(num_epochs=10, save_dir='./checkpoints')



def main():
    """Main pipeline."""
    print("="*70)
    print("Code Review Retrieval System - Complete Pipeline")
    print("="*70)
    
    # Check raw data
    if not os.path.exists("../your_full_database.jsonl"):
        print("Error: code_review_data.jsonl not found")
        print("Please prepare your raw data file in JSONL format, one JSON object per line, containing 'hunk', 'comment', and 'lang' fields")
        return
    
    # Step 1: Split dataset
    # If data_train.jsonl, data_val.jsonl, and data_test.jsonl already exist, skip splitting
    data_files = ["data_train.jsonl", "data_val.jsonl", "data_test.jsonl"]
    if all(os.path.exists(f) for f in data_files):
        print("Existing dataset files detected, skipping dataset splitting.")
        train_size = sum(1 for _ in open("data_train.jsonl"))
        val_size = sum(1 for _ in open("data_val.jsonl"))
        test_size = sum(1 for _ in open("data_test.jsonl"))
    else:
        train_size, val_size, test_size = split_dataset("../your_full_database.jsonl")
    
    print(f"\nTotal: {train_size + val_size + test_size} samples")
    
    # Step 2: Build triplets
    build_triplets_for_all_splits()
    
    # Step 3: Train model
    train_model()
    
    print("\n" + "="*70)
    print("Pipeline completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  - data_train.jsonl, data_val.jsonl, data_test.jsonl")
    print("  - triplets_train.jsonl, triplets_val.jsonl, triplets_test.jsonl")
    print("  - checkpoints/best_model.pt")
    print("  - code_review.index, indexed_data.jsonl")
    print("\nUsage:")
    print("  1. Load model: model.load_state_dict(torch.load('checkpoints/best_model.pt'))")
    print("  2. Initialize retriever: retriever = CodeReviewRetriever(model, tokenizer, device)")
    print("  3. Load index: retriever.load_index('code_review.index', 'indexed_data.jsonl')")
    print("  4. Retrieve: results = retriever.retrieve(query, top_k=10)")


if __name__ == "__main__":
    main()