import json
import re

FILES = {
    "Llama-3.1-8B-Instruct": "../results/llama_3.1_8b_instruct_ROME_120.json",
    "Gemma-3-12B-Instruct":  "../results/gemma_3_12b_it_ROME_120.json",
}

def clean_prompt(prompt: str) -> str:
    # JSON-parsed strings may contain doubled double-quotes (""word"" → "word")
    # from the original CSV-style escaping
    return re.sub(r'""([^""]+)""', r'"\1"', prompt).strip('"')

output = []

for model_name, filepath in FILES.items():
    with open(filepath) as f:
        data = json.load(f)

    for entry in data:
        question     = clean_prompt(entry["requested_edit"]["prompt"])
        model_resp   = entry["pre"]["edit_output"][0]
        model_resp_post = entry["post"]["edit_output"][0]
        hallucinated = entry["pre"]["edit_acc"][0] == 0   # 0 = wrong answer = hallucination
        hallucinated_post_edit = entry["post"]["edit_acc"][0] == 0

        record = {
            "model": model_name,
            "question": question,
            "model_response": model_resp,
            "model_response_post_edit": model_resp_post,
            "hallucination": hallucinated,
            "hallucination_post_edit": hallucinated_post_edit,
        }
        output.append(record)

with open("../results/combined_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Written {len(output)} records ({len(output)//2} per model).")
