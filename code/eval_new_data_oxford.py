import os
from util import *
import transformers
import pandas as pd
from tqdm import tqdm
from hallucination_editor import system_msg_eval


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_id_format = model_id.split('/')[-1].replace('-', '_').lower()

data_path = "../data/dataset_sample_with_answers.json"

tok_qa = transformers.AutoTokenizer.from_pretrained(model_id)
model_qa = transformers.AutoModelForCausalLM.from_pretrained(model_id).to('cuda:0')

# model_id_eval = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_eval = transformers.AutoModelForCausalLM.from_pretrained(model_id_eval, torch_dtype='auto').to('cuda:1')
# tok_eval = transformers.AutoTokenizer.from_pretrained(model_id_eval)


def get_response(model, tok, messages, max_new_tokens=1): 
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(model.device)
    output_ids = model.generate(**msg_tokenized, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(output_ids[0][msg_tokenized['input_ids'].shape[-1]:], skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')


def evaluate_responses(model_eval, tok_eval, df, system_msg_eval, user_msg_eval_template="Text 1: {label} \nText 2: {output_qa}"):
    for i in df.index:
        label = df.loc[i, 'object']
        output_qa = df.loc[i, f"output_{model_id_format}"]
        eval_res = 0

        if output_qa.lower() in label.lower() or label.lower() in output_qa.lower() or 'unknown' in output_qa.lower():  # Rule-based fuzzy match
            eval_res = 1
            if output_qa.lower() == label.lower():
                print(f"Label: {label:<35} Prediction: {output_qa:<35} Evaluation: Exact Match")
            else:
                print(f"Label: {label:<35} Prediction: {output_qa:<35} Evaluation: Partial Match")
        else:
            user_msg_eval = user_msg_eval_template.format(label=label, output_qa=output_qa)
            messages_eval = [{"role": "system", "content": system_msg_eval}, {"role": "user", "content": user_msg_eval}]
            response_eval = get_response(model_eval, tok_eval, messages_eval)
            if response_eval != '0':
                print(f"Label: {label:<35} Prediction: {output_qa:<35} Evaluation: Semantic Match")
                eval_res = 1
                
        df.loc[i, f"eval_{model_id_format}"] = eval_res
    hallu_count = df[df[f'eval_{model_id_format}']==0].shape
    print(f"Hallucination ratio: {hallu_count[0]/len(df)} df_hallucination.shape: {hallu_count}")
    return df

