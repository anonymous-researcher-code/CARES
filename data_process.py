import json
import re

def extract_patch_msg_lang(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            diff_text = data.get("patch", "")
            new_code = extract_new_code_from_diff(diff_text)
            diff = re.sub(r'^@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@\s*', '', diff_text)
            item = {
                "diff": diff,
                "new": new_code,
                "msg": data.get("msg", "")
            }
            if "lang" in data:
                item["lang"] = data["lang"]
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_file_path1 = "../original_dataset.jsonl" 
    output_file_path = "../dataset.jsonl"
    extract_patch_msg_lang(input_file_path, output_file_path)