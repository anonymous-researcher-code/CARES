"""
UniXcoder contrastive fine-tuning trainer - complete implementation.

Includes:
1. UniXcoderContrastiveModel - model architecture
2. ContrastiveDataset - dataset
3. ContrastiveLoss - loss function
4. ContrastiveTrainer - trainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import json
import numpy as np


# ============================================================
# 1. Model definition
# ============================================================

class UniXcoderContrastiveModel(nn.Module):
    """
    Contrastive learning model based on UniXcoder.
    
    Architecture: UniXcoder encoder + projection layer.
    """
    
    def __init__(self, model_name: str = "microsoft/unixcoder-base-nine", 
                 projection_dim: int = 256):
        """
        Args:
            model_name: Name of the pretrained UniXcoder model.
            projection_dim: Projected dimension (final embedding dimension).
        """
        super().__init__()
        
        # Load pretrained UniXcoder encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768
        
        # Projection layer: 768 -> projection_dim
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        print(f"✓ Model initialization completed")
        print(f"  Encoder: {model_name}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Projection dimension: {projection_dim}")
    
    def encode(self, input_ids, attention_mask):
        """
        Encode input into embedding vectors.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
        
        Returns:
            Normalized embeddings, shape (batch_size, projection_dim)
        """
        # Pass through encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token (first position)
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # Pass through projection layer
        projected = self.projector(pooled)
        
        # L2 normalization
        normalized = F.normalize(projected, p=2, dim=1)
        
        return normalized
    
    def forward(self, anchor_ids, anchor_mask, 
                positive_ids, positive_mask,
                negative_ids, negative_mask):
        """
        Forward pass: encode anchor, positive, and negative.
        
        Returns:
            Three embeddings: (anchor_emb, positive_emb, negative_emb)
        """
        anchor_emb = self.encode(anchor_ids, anchor_mask)
        positive_emb = self.encode(positive_ids, positive_mask)
        negative_emb = self.encode(negative_ids, negative_mask)
        
        return anchor_emb, positive_emb, negative_emb


# ============================================================
# 2. Dataset definition
# ============================================================

class ContrastiveDataset(Dataset):
    """
    Contrastive learning dataset.
    
    Load data from triplet files and perform tokenization.
    """
    
    def __init__(self, triplets_file: str, tokenizer, max_length: int = 512):
        """
        Args:
            triplets_file: Path to triplet data file.
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.triplets = []
        
        # Load triplet data
        print(f"Loading dataset: {triplets_file}")
        with open(triplets_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.triplets.append(json.loads(line))
        
        print(f"✓ Loaded {len(self.triplets)} training samples")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """
        Get the training sample at index idx.
        
        Returns:
            A dictionary containing anchor, positive, and negative inputs.
        """
        triplet = self.triplets[idx]
        
        # # Combine comment and diff as input text
        # anchor_text = f"{triplet['anchor_comment']} [SEP] {triplet['anchor_diff']}"
        # positive_text = f"{triplet['positive_comment']} [SEP] {triplet['positive_diff']}"
        # Use diff as input text
        anchor_text = f"{triplet['anchor_diff']}"
        positive_text = f"{triplet['positive_diff']}"

        # Randomly select one negative sample
        neg_idx = np.random.randint(len(triplet['negative_diffs']))
        negative_text = f"{triplet['negative_diffs'][neg_idx]}"
        
        # Tokenize
        anchor_enc = self.tokenizer(
            anchor_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        positive_enc = self.tokenizer(
            positive_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        negative_enc = self.tokenizer(
            negative_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'anchor_input_ids': anchor_enc['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_enc['attention_mask'].squeeze(0),
            'positive_input_ids': positive_enc['input_ids'].squeeze(0),
            'positive_attention_mask': positive_enc['attention_mask'].squeeze(0),
            'negative_input_ids': negative_enc['input_ids'].squeeze(0),
            'negative_attention_mask': negative_enc['attention_mask'].squeeze(0),
        }


# ============================================================
# 3. Loss function definition
# ============================================================

class ContrastiveLoss(nn.Module):
    """
    Contrastive learning loss function.
    
    Combines Triplet Loss and InfoNCE Loss.
    """
    
    def __init__(self, margin: float = 0.5, temperature: float = 0.07):
        """
        Args:
            margin: Margin for triplet loss.
            temperature: Temperature parameter for InfoNCE.
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def triplet_loss(self, anchor, positive, negative):
        """
        Triplet loss.
        
        Goal: make d(anchor, positive) + margin < d(anchor, negative).
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    
    def infonce_loss(self, anchor, positive, negative):
        """
        InfoNCE loss (standard contrastive learning loss).
        
        Distinguish positive samples from negative samples.
        """
        # Compute similarity (already normalized, so cosine similarity = inner product)
        pos_sim = F.cosine_similarity(anchor, positive) / self.temperature
        neg_sim = F.cosine_similarity(anchor, negative) / self.temperature
        
        # Build a classification problem: positive samples should score higher
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def forward(self, anchor, positive, negative):
        """
        Combined loss.
        
        Returns:
            Total loss = 0.5 * triplet_loss + 0.5 * infonce_loss
        """
        triplet = self.triplet_loss(anchor, positive, negative)
        infonce = self.infonce_loss(anchor, positive, negative)
        
        return 0.5 * triplet + 0.5 * infonce


# ============================================================
# 4. Trainer definition
# ============================================================

class ContrastiveTrainer:
    """
    Contrastive learning trainer.
    
    Manages training loops, validation, and model saving.
    """
    
    def __init__(self, model, train_loader, val_loader, 
                 device='cuda', lr=2e-5):
        """
        Args:
            model: UniXcoderContrastiveModel instance.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            device: Training device.
            lr: Learning rate.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = ContrastiveLoss(margin=0.5, temperature=0.07)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * 10  # Assume 10 epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        print(f"✓ Trainer initialization completed")
        print(f"  Device: {device}")
        print(f"  Learning rate: {lr}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
    
    def train_epoch(self):
        """
        Train one epoch.
        
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move data to device
            anchor_ids = batch['anchor_input_ids'].to(self.device)
            anchor_mask = batch['anchor_attention_mask'].to(self.device)
            positive_ids = batch['positive_input_ids'].to(self.device)
            positive_mask = batch['positive_attention_mask'].to(self.device)
            negative_ids = batch['negative_input_ids'].to(self.device)
            negative_mask = batch['negative_attention_mask'].to(self.device)
            
            # Forward pass
            anchor_emb, positive_emb, negative_emb = self.model(
                anchor_ids, anchor_mask,
                positive_ids, positive_mask,
                negative_ids, negative_mask
            )
            
            # Compute loss
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item() 
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """
        Evaluate on the validation set.
        
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                anchor_ids = batch['anchor_input_ids'].to(self.device)
                anchor_mask = batch['anchor_attention_mask'].to(self.device)
                positive_ids = batch['positive_input_ids'].to(self.device)
                positive_mask = batch['positive_attention_mask'].to(self.device)
                negative_ids = batch['negative_input_ids'].to(self.device)
                negative_mask = batch['negative_attention_mask'].to(self.device)
                
                anchor_emb, positive_emb, negative_emb = self.model(
                    anchor_ids, anchor_mask,
                    positive_ids, positive_mask,
                    negative_ids, negative_mask
                )
                
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int, save_dir: str):
        """
        Full training pipeline.
        
        Args:
            num_epochs: Number of training epochs.
            save_dir: Model save directory.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print("\n" + "="*60)
        print("Training started")
        print("="*60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                checkpoint_path = f'{save_dir}/best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, checkpoint_path)
                
                print(f"✓ Best model saved to {checkpoint_path}")
                print(f"  Val Loss: {val_loss:.4f}")
        
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("="*60)


if __name__ == "__main__":
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    
    # Prepare data
    train_dataset = ContrastiveDataset('triplets_train.jsonl', tokenizer)
    val_dataset = ContrastiveDataset('triplets_val.jsonl', tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize model
    model = UniXcoderContrastiveModel(
        model_name="microsoft/unixcoder-base",
        projection_dim=256
    )
    
    # Print model parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total params: {total_params / 1e6:.2f}M")
    print(f"  Trainable params: {trainable_params / 1e6:.2f}M")
    
    # Train
    trainer = ContrastiveTrainer(
        model, 
        train_loader, 
        val_loader, 
        device=device,
        lr=2e-5
    )
    
    trainer.train(num_epochs=10, save_dir='./checkpoints')