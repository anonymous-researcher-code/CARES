import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
import shutil
from tqdm import tqdm
from transformers.utils import logging
import warnings
import json
from nltk.translate import bleu_score
import statistics
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, RobertaTokenizer

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


model_name = "CodeReviewer" 
# Choose strategy among "rag_pair", , "vanilla"
rag_strategy = "rag_pair" # here to modify

dataset_base = "./dataset_path/"
retrieval_base = "../rag_candidates"
output_ckp_base = '../output_finetuned_checkpoints/'
path_train = dataset_base+'train.jsonl'
path_val = dataset_base+'valid.jsonl'
total_topk = 30

if rag_strategy == "rag_pair":
    top_k =  4 
else:
    top_k = 30

batch_size = 12
max_input_length=2048
max_target_length=128
num_beams = 10


if model_name == "CodeReviewer":
    model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/codereviewer")
    tokenizer =  AutoTokenizer.from_pretrained("microsoft/codereviewer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on: {device}")

def processDataset(dataset):
    code_list, comment_list = [], []
    for data in dataset:
        code = data.get('diff', '')
        comment = data.get('msg', '')
        code_list.append(code)
        comment_list.append(comment)
    return code_list, comment_list


with open(path_train, "r", encoding="utf-8") as f:
    train_dataset = [json.loads(line) for line in f]

with open(path_val, "r", encoding="utf-8") as f:
    val_dataset = [json.loads(line) for line in f]

train_code, train_comment = processDataset(train_dataset)
val_code, val_comment = processDataset(val_dataset)

def get_topk_candidates(topk, total_topk, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        file_lines = f.read().splitlines()

    num_instances = len(file_lines) // total_topk
    retrieved_file = [
    file_lines[i * total_topk:(i + 1) * total_topk][:topk] for i in range(num_instances)
    ]

    return retrieved_file

train_top30_candidate_comment_file = os.path.join(retrieval_base, f"train_to_train_retrieval_top30_comment.txt")
train_top30_candidate_code_file = os.path.join(retrieval_base, f"train_to_train_retrieval_top30_code.txt")
val_top30_candidate_comment_file = os.path.join(retrieval_base, f"val_to_train_retrieval_top30_comment.txt")
val_top30_candidate_code_file = os.path.join(retrieval_base, f"val_to_train_retrieval_top30_code.txt")

train_candidiate_comment = get_topk_candidates(top_k, total_topk, train_top30_candidate_comment_file)
train_candidiate_code = get_topk_candidates(top_k, total_topk, train_top30_candidate_code_file)
val_candidiate_comment = get_topk_candidates(top_k, total_topk, val_top30_candidate_comment_file)
val_candidiate_code = get_topk_candidates(top_k, total_topk, val_top30_candidate_code_file)

def build_rag_inputs(rag_strategy, inputs, candidate_comment, candidate_code):
    rag_inputs = []
    for i in tqdm(range(len(inputs)), desc="Building RAG inputs"):
        x = inputs[i]
        topk_codes = candidate_code[i]
        topk_comments = candidate_comment[i]
        if rag_strategy == "rag_pair":
            for j in range(len(topk_codes)):
                x += "[nsep]" + topk_comments[j] + "[csep]" + topk_codes[j]
        rag_inputs.append(x)
    return rag_inputs

train_rag_input = build_rag_inputs(rag_strategy, train_code, train_candidiate_comment, train_candidiate_code)
val_rag_input = build_rag_inputs(rag_strategy, val_code, val_candidiate_comment, val_candidiate_code)
train_target = train_comment
val_target = val_comment


class FineTuneDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_input_length=512, max_target_length=128):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        source_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        source_enc = {k: v.squeeze(0) for k, v in source_enc.items()}
        target_ids = target_enc["input_ids"].squeeze(0)
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        return source_enc, target_ids
    
train_dataset = FineTuneDataset(train_rag_input, train_target, tokenizer, max_input_length, max_target_length)
val_dataset = FineTuneDataset(val_rag_input, val_target, tokenizer, max_input_length, max_target_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


chencherry = bleu_score.SmoothingFunction()

def validate_model():
    model.eval()
    perfect_predictions = 0
    BLEUscore = []
    total_samples = len(val_dataset)

    outputs, targets = [], []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            inputs, target = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            target = target.to(device) 

            generated_ids = model.generate(
                **inputs,
                max_length=max_target_length,
                num_beams=num_beams, 
                num_return_sequences=1,  
                early_stopping=True
            )

            pred_texts = [tokenizer.decode(gid, skip_special_tokens=True) for gid in generated_ids]

            target_texts = [
                tokenizer.decode(t[t >= 0].tolist(), skip_special_tokens=True) for t in target
            ]

            outputs.extend(pred_texts)
            targets.extend(target_texts)
    
            for pred, target in zip(pred_texts, target_texts):
                if " ".join(pred.split()) == " ".join(target.split()):
                    perfect_predictions += 1
                BLEUscore.append(bleu_score.sentence_bleu([target], pred, smoothing_function=chencherry.method1))
                
    pp_percentage = (perfect_predictions * 100) / total_samples
    print(f'Perfect Prediction (PP): {perfect_predictions}/{total_samples} ({pp_percentage:.2f}%)')
    print('BLEU mean:', statistics.mean(BLEUscore))

    return perfect_predictions

def save_best_ckp():
    best_ckp_path = os.path.join(output_ckp_base, model_name + "_" + rag_strategy + f"_finetuned_best_ckp_{best_ckp_epoch}")
    model.save_pretrained(best_ckp_path)
    tokenizer.save_pretrained(best_ckp_path)
    print(f"Saved best_ckp model for epoch {best_ckp_epoch}")

def save_ckp(epoch):
    ckp_path = os.path.join(output_ckp_base, model_name + "_" + rag_strategy + f"_finetuned_ckp_{epoch}")
    model.save_pretrained(ckp_path)
    tokenizer.save_pretrained(ckp_path)
    print(f"Saved ckp model for epoch {epoch}")

def remove_best_ckp():
    if best_pp == 0:
        return
    best_ckp_path = os.path.join(output_ckp_base, model_name + "_" + rag_strategy + f"_finetuned_best_ckp_{best_ckp_epoch}")
    shutil.rmtree(best_ckp_path)
    print(f"Removed best_ckp model for epoch {best_ckp_epoch}")


start_epoch = 1
end_epoch = 20
patience = 3
best_pp = 0
best_ckp_epoch = 0
epochs_since_improvement = 0 

learning_rate = 3e-5
accumulation_steps = 3 
weight_decay = 0.01
warmup_ratio = 0.1
max_grad_norm = 1.0

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
total_steps = len(train_dataloader) * (end_epoch - start_epoch + 1) // accumulation_steps
warmup_steps = int(total_steps * warmup_ratio)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

scaler = GradScaler() 

model.train()
for epoch in range(start_epoch, end_epoch + 1):
    epoch_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{end_epoch}")):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        with autocast(): 
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()
        epoch_loss += loss.item() * accumulation_steps

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch}/{end_epoch}, Loss: {avg_loss:.4f}")

    save_ckp(epoch)
    curr_pp = validate_model()

    if curr_pp > best_pp:
        remove_best_ckp()
        best_pp = curr_pp
        best_ckp_epoch = epoch
        save_best_ckp()
        epochs_since_improvement = 0
        print(f"New best PP: {best_pp} at epoch {best_ckp_epoch}")
    else:
        epochs_since_improvement += 1
        print(f"No improvement for {epochs_since_improvement} epoch(s).")

    if epochs_since_improvement >= patience:
        print(f"Early stopping triggered after {patience} epochs without improvement.")
        break

