import os, warnings
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.translate import bleu_score
from tqdm import tqdm
import statistics
from transformers.utils import logging
import warnings
import json

from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, RobertaTokenizer

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

model_name = "CodeReviewer"

# Choose strategy among "rag_pair", "vanilla"
rag_strategy = "rag_pair" # here to modify

# specify the path of fune-tuned model checkpoint to be evaluated. ex) './output/fine_tuned_checkpoints/CodeT5_rag_pair_finetuned_best_ckp_2'
path_fine_tuned_ckp = './RAG-codereviewer-finetune/code/fine-tuning/output_finetuned_checkpoints/..' # here to modify

# specify the path where inference output to be loaded. ex) "./output/inference/"
path_inference_output = "../" # here to modify

dataset_base = "../CodeReview_dataset/"
retrieval_base = "../rag_code_candidate/"
output_ckp_base = '../output_finetuned_checkpoints/'
path_test = dataset_base+'test.jsonl'
total_topk = 30

if rag_strategy == "rag_pair":
    top_k =  8 
else:
    top_k = 30

batch_size = 12
max_input_length=512
max_target_length=128
num_beams = 10



model_name == "CodeReviewer":
model = AutoModelForSeq2SeqLM.from_pretrained(path_fine_tuned_ckp)
tokenizer =  AutoTokenizer.from_pretrained(path_fine_tuned_ckp)

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

with open(path_test, "r", encoding="utf-8") as f:
    # 逐行读取jsonl，每行为一个json对象
    test_dataset = [json.loads(line) for line in f]

test_code, test_comment = processDataset(test_dataset)

def get_topk_candidates(topk, total_topk, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        file_lines = f.read().splitlines()

    num_instances = len(file_lines) // total_topk
    retrieved_file = [
    file_lines[i * total_topk:(i + 1) * total_topk][:topk] for i in range(num_instances)
    ]
    
    return retrieved_file

test_top30_candidate_comment_file = os.path.join(retrieval_base, f"test_to_train_retrieval_top30_comment.txt")
test_top30_candidate_code_file = os.path.join(retrieval_base, f"test_to_train_retrieval_top30_code.txt")

test_candidate_comment = get_topk_candidates(top_k, total_topk, test_top30_candidate_comment_file)
test_candidate_code = get_topk_candidates(top_k, total_topk, test_top30_candidate_code_file)

def build_rag_inputs(rag_strategy, inputs, candidate_comment, candidate_code):
    rag_inputs = []
    for i in tqdm(range(len(inputs)), desc="Building RAG inputs"):
        x = inputs[i]
        topk_codes = candidate_code[i]
        topk_comments = candidate_comment[i]
        if rag_strategy == "rag_singleton":
            for comment in topk_comments:
                x += "[nsep]" + comment
        elif rag_strategy == "rag_pair":
            for j in range(len(topk_codes)):
                x += "[nsep]" + topk_comments[j] + "[csep]" + topk_codes[j]
        rag_inputs.append(x)
    return rag_inputs

test_rag_input = build_rag_inputs(rag_strategy, test_code, test_candidate_comment, test_candidate_code)
test_target = test_comment

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
    
test_dataset = FineTuneDataset(test_rag_input, test_target, tokenizer, max_input_length, max_target_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

chencherry = bleu_score.SmoothingFunction()

def test_model():
    model.eval()
    perfect_predictions = 0
    BLEUscore = []
    total_samples = len(test_dataset)

    outputs, targets = [], []
    prediction_path = os.path.join(path_inference_output, model_name + '_' + rag_strategy + '_predictions.txt')

    with torch.no_grad(), open(prediction_path, 'w', encoding='utf-8') as f_out:
        for batch in tqdm(test_dataloader, desc="Testing"):
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

            for pred, tgt in zip(pred_texts, target_texts):
                f_out.write(pred + '\n')

                if " ".join(pred.split()) == " ".join(tgt.split()):
                    perfect_predictions += 1
                BLEUscore.append(
                    bleu_score.sentence_bleu([tgt], pred, smoothing_function=chencherry.method1)
                )

    pp_percentage = (perfect_predictions * 100) / total_samples
    print(f'Perfect Prediction (PP): {perfect_predictions}/{total_samples} ({pp_percentage:.2f}%)')
    print('BLEU mean:', statistics.mean(BLEUscore))
    print(f'Predictions written to {prediction_path}')

if __name__ == "__main__":
    test_model() 