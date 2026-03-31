"""
unixcoder_contrastive_trainer.py

UniXcoder对比学习微调训练器 - 完整实现

包含：
1. UniXcoderContrastiveModel - 模型架构
2. ContrastiveDataset - 数据集
3. ContrastiveLoss - 损失函数
4. ContrastiveTrainer - 训练器
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
# 1. 模型定义
# ============================================================

class UniXcoderContrastiveModel(nn.Module):
    """
    基于UniXcoder的对比学习模型
    
    架构：UniXcoder Encoder + 投影层
    """
    
    def __init__(self, model_name: str = "microsoft/unixcoder-base-nine", 
                 projection_dim: int = 256):
        """
        Args:
            model_name: UniXcoder预训练模型名称
            projection_dim: 投影后的维度（最终embedding维度）
        """
        super().__init__()
        
        # 加载预训练的UniXcoder编码器
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768
        
        # 投影层：768 -> projection_dim
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        print(f"✓ 模型初始化完成")
        print(f"  编码器: {model_name}")
        print(f"  隐藏层维度: {self.hidden_size}")
        print(f"  投影维度: {projection_dim}")
    
    def encode(self, input_ids, attention_mask):
        """
        编码输入为embedding向量
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: 注意力掩码, shape (batch_size, seq_len)
        
        Returns:
            归一化的embedding, shape (batch_size, projection_dim)
        """
        # 通过编码器
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 提取[CLS] token（第一个位置）
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # 通过投影层
        projected = self.projector(pooled)
        
        # L2归一化
        normalized = F.normalize(projected, p=2, dim=1)
        
        return normalized
    
    def forward(self, anchor_ids, anchor_mask, 
                positive_ids, positive_mask,
                negative_ids, negative_mask):
        """
        前向传播：编码anchor、positive、negative
        
        Returns:
            三个embedding: (anchor_emb, positive_emb, negative_emb)
        """
        anchor_emb = self.encode(anchor_ids, anchor_mask)
        positive_emb = self.encode(positive_ids, positive_mask)
        negative_emb = self.encode(negative_ids, negative_mask)
        
        return anchor_emb, positive_emb, negative_emb


# ============================================================
# 2. 数据集定义
# ============================================================

class ContrastiveDataset(Dataset):
    """
    对比学习数据集
    
    从三元组文件加载数据并进行tokenization
    """
    
    def __init__(self, triplets_file: str, tokenizer, max_length: int = 512):
        """
        Args:
            triplets_file: 三元组数据文件路径
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.triplets = []
        
        # 加载三元组数据
        print(f"加载数据集: {triplets_file}")
        with open(triplets_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.triplets.append(json.loads(line))
        
        print(f"✓ 加载了 {len(self.triplets)} 个训练样本")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """
        获取第idx个训练样本
        
        Returns:
            包含anchor、positive、negative输入的字典
        """
        triplet = self.triplets[idx]
        
        # # 组合comment和diff作为输入文本
        # anchor_text = f"{triplet['anchor_comment']} [SEP] {triplet['anchor_diff']}"
        # positive_text = f"{triplet['positive_comment']} [SEP] {triplet['positive_diff']}"
        # diff作为输入文本
        anchor_text = f"{triplet['anchor_diff']}"
        positive_text = f"{triplet['positive_diff']}"

        # 随机选择一个负样本
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
# 3. 损失函数定义
# ============================================================

class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数
    
    组合Triplet Loss和InfoNCE Loss
    """
    
    def __init__(self, margin: float = 0.5, temperature: float = 0.07):
        """
        Args:
            margin: Triplet loss的边界
            temperature: InfoNCE的温度参数
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def triplet_loss(self, anchor, positive, negative):
        """
        三元组损失
        
        目标：使 d(anchor, positive) + margin < d(anchor, negative)
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    
    def infonce_loss(self, anchor, positive, negative):
        """
        InfoNCE损失（对比学习标准损失）
        
        将正样本从负样本中区分出来
        """
        # 计算相似度（已归一化，所以余弦相似度=内积）
        pos_sim = F.cosine_similarity(anchor, positive) / self.temperature
        neg_sim = F.cosine_similarity(anchor, negative) / self.temperature
        
        # 构造分类问题：正样本应该得分更高
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def forward(self, anchor, positive, negative):
        """
        组合损失
        
        Returns:
            总损失 = 0.5 * triplet_loss + 0.5 * infonce_loss
        """
        triplet = self.triplet_loss(anchor, positive, negative)
        infonce = self.infonce_loss(anchor, positive, negative)
        
        return 0.5 * triplet + 0.5 * infonce


# ============================================================
# 4. 训练器定义
# ============================================================

class ContrastiveTrainer:
    """
    对比学习训练器
    
    管理训练循环、验证、模型保存
    """
    
    def __init__(self, model, train_loader, val_loader, 
                 device='cuda', lr=2e-5):
        """
        Args:
            model: UniXcoderContrastiveModel实例
            train_loader: 训练数据DataLoader
            val_loader: 验证数据DataLoader
            device: 训练设备
            lr: 学习率
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 损失函数
        self.criterion = ContrastiveLoss(margin=0.5, temperature=0.07)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=0.01
        )
        
        # 学习率调度器
        total_steps = len(train_loader) * 10  # 假设10个epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        print(f"✓ 训练器初始化完成")
        print(f"  设备: {device}")
        print(f"  学习率: {lr}")
        print(f"  训练样本数: {len(train_loader.dataset)}")
        print(f"  验证样本数: {len(val_loader.dataset)}")
    
    def train_epoch(self):
        """
        训练一个epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # 移动数据到设备
            anchor_ids = batch['anchor_input_ids'].to(self.device)
            anchor_mask = batch['anchor_attention_mask'].to(self.device)
            positive_ids = batch['positive_input_ids'].to(self.device)
            positive_mask = batch['positive_attention_mask'].to(self.device)
            negative_ids = batch['negative_input_ids'].to(self.device)
            negative_mask = batch['negative_attention_mask'].to(self.device)
            
            # 前向传播
            anchor_emb, positive_emb, negative_emb = self.model(
                anchor_ids, anchor_mask,
                positive_ids, positive_mask,
                negative_ids, negative_mask
            )
            
            # 计算损失
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录
            total_loss += loss.item() 
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """
        在验证集上评估
        
        Returns:
            平均验证损失
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
        完整训练流程
        
        Args:
            num_epochs: 训练轮数
            save_dir: 模型保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print("\n" + "="*60)
        print("开始训练")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        print("="*60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 打印结果
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存最佳模型
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
                
                print(f"✓ 保存最佳模型到 {checkpoint_path}")
                print(f"  Val Loss: {val_loss:.4f}")
        
        print("\n" + "="*60)
        print("训练完成！")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print("="*60)


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    
    # 准备数据
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
    
    # 初始化模型
    model = UniXcoderContrastiveModel(
        model_name="microsoft/unixcoder-base",
        projection_dim=256
    )
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"  总参数: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 训练
    trainer = ContrastiveTrainer(
        model, 
        train_loader, 
        val_loader, 
        device=device,
        lr=2e-5
    )
    
    trainer.train(num_epochs=10, save_dir='./checkpoints')