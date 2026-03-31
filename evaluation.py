import os
import json
import numpy as np
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score_func
from rouge_score import rouge_scorer
import evaluate
import nltk

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class CodeReviewEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {self.device}")
        self.chencherry = SmoothingFunction()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def load_lines(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    # ====================================================================
    # Retriever Quality Evaluation
    # ====================================================================
    def evaluate_retriever(self, candidates_file, ground_truth_file, top_k=4, batch_size=64):
        
        candidates_raw = self.load_lines(candidates_file)
        ground_truths = self.load_lines(ground_truth_file)

        # Check data alignment
        assert len(candidates_raw) == len(ground_truths) * top_k, \
            f"Data misalignment! Candidate lines ({len(candidates_raw)}) are not {top_k}x GT lines ({len(ground_truths)})."

        num_samples = len(ground_truths)
        print(f"Total samples: {num_samples} | Top-K: {top_k}")

        
        expanded_refs = []
        expanded_cands = []

        bleu_scores = []
        rouge_scores = []
        
        for i in tqdm(range(num_samples), desc="Computing traditional metrics (BLEU/ROUGE)"):
            gt = ground_truths[i]
            #  top_k  candidates
            current_cands = candidates_raw[i*top_k : (i+1)*top_k]
            
            expanded_refs.extend([gt] * top_k)
            expanded_cands.extend(current_cands)

            # --- BLEU  ---
            sample_bleus = []
            gt_tokens = gt.split()
            for cand in current_cands:
                cand_tokens = cand.split()
                b_score = sentence_bleu([gt_tokens], cand_tokens, smoothing_function=self.chencherry.method1)
                sample_bleus.append(b_score)
            bleu_scores.append(np.mean(sample_bleus))

            # --- ROUGE-L ---
            sample_rouges = []
            for cand in current_cands:
                r_score = self.rouge_scorer.score(gt, cand)['rougeL'].fmeasure
                sample_rouges.append(r_score)
            rouge_scores.append(np.mean(sample_rouges))

        # ---------------- 2. BERTScore ----------------
        P, R, F1 = bert_score_func(
            expanded_cands, 
            expanded_refs, 
            lang="en", 
            verbose=True, 
            device=self.device, 
            batch_size=batch_size,
            model_type="roberta-large"
        )
        
        f1_matrix = F1.reshape(num_samples, top_k)
        bert_scores = f1_matrix.mean(dim=1).tolist()
        max_bert_scores = f1_matrix.max(dim=1).values.tolist()

        # ---------------- 3. Semantic Recall ----------------
        threshold = 0.85
        hits = sum([1 for s in max_bert_scores if s > threshold])
        semantic_recall = hits / num_samples * 100

        results = {
            "BLEU": np.mean(bleu_scores),
            "ROUGE-L": np.mean(rouge_scores),
            "BERTScore": np.mean(bert_scores),
            f"Semantic-Recall@{threshold}": semantic_recall
        }
        
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        
        return results

    # ====================================================================
    # Generator Quality Evaluation
    # ====================================================================
    def evaluate_generator(self, predictions_file, ground_truth_file, batch_size=64):
        
        preds = self.load_lines(predictions_file)
        refs = self.load_lines(ground_truth_file)
        
        assert len(preds) == len(refs), f"Prediction count ({len(preds)}) does not match GT count ({len(refs)})!"
        
        # 1. BERTScore
        P, R, F1 = bert_score_func(
            preds, refs, lang="en", verbose=True, device=self.device, batch_size=batch_size, model_type="roberta-large"
        )
        avg_bert_score = F1.mean().item()

        # 2. BLEU-4 
        bleu_scores = []
        for p, r in zip(preds, refs):
            bleu_scores.append(sentence_bleu([r], p, smoothing_function=self.chencherry.method1))
        avg_bleu = np.mean(bleu_scores)

        # 3. ROUGE-L
        rouge_scores = []
        for p, r in zip(preds, refs):
            rouge_scores.append(self.rouge_scorer.score(r, p)['rougeL'].fmeasure)
        avg_rouge = np.mean(rouge_scores)

        results = {
            "BERTScore-F1": avg_bert_score,
            "BLEU-4": avg_bleu,
            "ROUGE-L": avg_rouge
        }

        print("\n>>> Generator evaluation results:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
            
        return results

    def evaluate_9lang(self, predictions_file, msg_test_jsonl, batch_size=64):
        '''
        Compute generator evaluation metrics for 9 languages using 'msg' and 'lang' fields from msg-test.jsonl.
        '''
        import collections

        # 1. Load predictions and lang/msg from msg-test.jsonl
        preds = self.load_lines(predictions_file)
        msg_lang_list = []
        with open(msg_test_jsonl, 'r', encoding='utf-8') as fin:
            for line in fin:
                if line.strip():
                    item = json.loads(line)
                    lang = item.get('lang', 'unknown')
                    msg = item.get('msg', '').strip()
                    msg_lang_list.append((lang, msg))

        assert len(preds) == len(msg_lang_list), \
            f"Prediction count ({len(preds)}) does not match msg-test.jsonl sample count ({len(msg_lang_list)})!"

        lang_to_preds = collections.defaultdict(list)
        lang_to_refs = collections.defaultdict(list)
        for pred, (lang, ref) in zip(preds, msg_lang_list):
            lang_to_preds[lang].append(pred)
            lang_to_refs[lang].append(ref)

        # Find all languages
        all_langs = sorted(lang_to_preds.keys())
        print(f"Detected languages: {all_langs}")

        results_per_lang = {}
        for lang in all_langs:
            print(f"\n--- {lang} ---")
            lang_preds = lang_to_preds[lang]
            lang_refs = lang_to_refs[lang]
            count = len(lang_preds)
            if count == 0:
                print(f"No samples for this language, skipped")
                continue

            # 1. BERTScore
            print(f"Computing BERTScore on {lang} ({count} samples)...")
            try:
                P, R, F1 = bert_score_func(lang_preds, lang_refs, lang="en", verbose=True, device=self.device, batch_size=batch_size, model_type="roberta-large")
                avg_bert_score = F1.mean().item()
            except Exception as e:
                print(f"  > BERTScore computation failed: {e}")
                avg_bert_score = -1

            # 2. BLEU
            print(f"Computing BLEU-4 on {lang} ...")
            bleu_scores = []
            for p, r in zip(lang_preds, lang_refs):
                bleu_scores.append(sentence_bleu([r], p, smoothing_function=self.chencherry.method1))
            avg_bleu = np.mean(bleu_scores) if bleu_scores else -1

            # 3. ROUGE-L
            print(f"Computing ROUGE-L on {lang} ...")
            rouge_scores = []
            for p, r in zip(lang_preds, lang_refs):
                try:
                    rouge_scores.append(self.rouge_scorer.score(r, p)['rougeL'].fmeasure)
                except Exception as e:
                    print(f"ROUGE computation exception: {e}")
            avg_rouge = np.mean(rouge_scores) if rouge_scores else -1

            # 4. Length
            avg_len = np.mean([len(p.split()) for p in lang_preds]) if lang_preds else -1

            results_per_lang[lang] = {
                "BERTScore-F1": avg_bert_score,
                "BLEU-4": avg_bleu,
                "ROUGE-L": avg_rouge,
                "Avg-Length": avg_len,
                "Count": count
            }

            print(f"[{lang}] Results: BERTScore-F1={avg_bert_score:.4f}  BLEU-4={avg_bleu:.4f}  ROUGE-L={avg_rouge:.4f}  Avg-Length={avg_len:.2f} Count={count}")

        print("\n>>> Generator evaluation results by language:")
        for lang, m in results_per_lang.items():
            print(f"  {lang}\tBERTScore-F1={m['BERTScore-F1']:.4f} | BLEU-4={m['BLEU-4']:.4f} | ROUGE-L={m['ROUGE-L']:.4f} | Avg-Length={m['Avg-Length']:.2f} | N={m['Count']}")

        return results_per_lang

# ====================================================================
# Main entry
# ====================================================================
if __name__ == "__main__":
    test_FILE = "../test.jsonl"
    GT_TXT = "test_ground_truth.txt"
    if not os.path.exists(os.path.join(os.getcwd(), GT_TXT)):
        if os.path.exists(MSG_FILE):
            print(f"Extracting msg from {MSG_FILE} to generate {GT_TXT} ...")
            with open(MSG_FILE, 'r', encoding='utf-8') as infile, open(GT_TXT, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    if line.strip():
                        item = json.loads(line)
                        msg = item.get('msg', '').replace('\n', ' ').strip()
                        outfile.write(msg + "\n")
    else:
        print(f"{GT_TXT} already exists in current directory, skipping extraction.")
    
    # Prediction files are under this directory
    BASE_DIR = "../output_test_pred_results/" 
    
    RETRIEVER_GT = os.path.join("test_ground_truth.txt")
    RETRIEVER_CANDS_VANILLA = os.path.join("../test_to_train_retrieval_top30_comment.txt")
    RETRIEVER_CANDS_unixcoder = os.path.join("../test_to_train_retrieval_top30_comment.txt")
    RETRIEVER_codeT5 = os.path.join("../test_to_train_retrieval_top30_comment.txt")


    GENERATOR_GT = os.path.join("test_ground_truth.txt")
    PRED_vanilla_codereviewer = os.path.join(BASE_DIR, "../generated_predictions.txt")
    PRED_contrastiveUnixcoder_codereviewer = os.path.join(BASE_DIR, "../generated_predictions.txt")
    PRED_VANILLA_LLAMA3 = os.path.join(BASE_DIR, "../generated_predictions.txt")
    PRED_contrastiveUnixcoder_llama3_direct_inference = os.path.join(BASE_DIR, "../generated_predictions.txt")  
    PRED_contrastiveUnixcoder_llama3_finetuned = os.path.join(BASE_DIR, "../generated_predictions.txt")



    evaluator = CodeReviewEvaluator()

    # --- Evaluate retriever ---
    if os.path.exists(RETRIEVER_CANDS_OURS):
        evaluator.evaluate_retriever(RETRIEVER_CANDS_OURS, RETRIEVER_GT)    
    if os.path.exists(RETRIEVER_CANDS_unixcoder):
        evaluator.evaluate_retriever(RETRIEVER_CANDS_unixcoder, RETRIEVER_GT)
    if os.path.exists(RETRIEVER_codeT5):
        evaluator.evaluate_retriever(RETRIEVER_codeT5, RETRIEVER_GT)

    # --- Evaluate generator ---
    models_to_test = {
        "CodeReviewer ": PRED_vanilla_codereviewer,
        "RAG-CodeReviewer": PRED_contrastiveUnixcoder_codereviewer,
        "LLaMA_Reviewer": PRED_VANILLA_LLAMA3
        "LLM-ICL": PRED_contrastiveUnixcoder_llama3_direct_inference
        "CARES": PRED_contrastiveUnixcoder_llama3_finetuned
    }


    for model_name, pred_file in models_to_test.items():
        if os.path.exists(pred_file):
            print(f"\n--- Evaluating model: {model_name} ---")
            evaluator.evaluate_generator(pred_file, GENERATOR_GT)


    for model_name, pred_file in models_to_test.items():
        if os.path.exists(pred_file) and os.path.exists(MSG_FILE):
            print(f"\n--- Evaluating model by language: {model_name} ---")
            evaluator.evaluate_9lang(pred_file, MSG_FILE)