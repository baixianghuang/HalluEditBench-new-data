import ast
import os
import gc
import json
import time
import torch
import argparse
import pandas as pd
from hallucination_editor import BaseEditor
from easyeditor import FTHyperParams, ROMEHyperParams
import openai


def parse_answer(answer):
    """Return a single canonical answer string from various answer formats."""
    if isinstance(answer, int) or isinstance(answer, float):
        return str(answer)
    if isinstance(answer, list):
        return str(answer[0])
    # String that looks like a Python list
    stripped = answer.strip()
    if stripped.startswith('['):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='llama3-8b')
    parser.add_argument('--data_size', default=None, type=int)
    parser.add_argument('--hparams_dir', default='./hparams', type=str)
    parser.add_argument('--results_dir', default='../results/', type=str)
    parser.add_argument('--edit_method', default=None, help='Edit method to use')
    parser.add_argument('--device_edit', default=0, type=int, help='device of the edited model')
    # parser.add_argument('--dataset_path', default='../data/dataset_sample_with_answers.json', type=str)
    parser.add_argument('--dataset_path', default='../data/dataset_ox_new.json', type=str)
    parser.add_argument('--overwrite_result', default=False, action='store_true', help='Overwrite the existing result file')
    parser.add_argument('--model_eval', default='gpt-4o-mini', help='Judge model: OpenAI model name (e.g. gpt-4o-mini) or local HuggingFace model id')
    args = parser.parse_args()
    start_time = time.time()

    editing_methods = ['FT-M', 'ROME']
    if args.edit_method is not None:
        editing_methods = [args.edit_method]

    # Load dataset
    with open(args.dataset_path, 'r') as f:
        data = json.load(f)

    size = len(data) if args.data_size is None else args.data_size

    subjects = [d['subject'] for d in data]
    questions = [d['question'] for d in data]
    targets = [parse_answer(d['answer']) for d in data]

    for editing_method in editing_methods:
        if editing_method in ['FT-M', 'FT-L']:
            editing_hparams = FTHyperParams
        elif editing_method == 'ROME':
            editing_hparams = ROMEHyperParams
        else:
            raise NotImplementedError(f"Unsupported editing method: {editing_method}")

        hparams = editing_hparams.from_hparams(f'{args.hparams_dir}/{editing_method}/{args.model_name}')
        model_id_format = hparams.model_name.split('/')[-1].replace('-', '_').lower()

        result_dir = f'{args.results_dir}'
        result_path = f'{result_dir}/{model_id_format}_{editing_method}_{size}.json'

        print(f'\nModel: {model_id_format}, Editing dataset_sample with {editing_method}...\n')
        if os.path.exists(result_path):
            print(f'Result {result_path} already exists')
            if args.overwrite_result:
                print('Overwriting...\n')
            else:
                continue

        hparams.device = args.device_edit
        editor = BaseEditor.from_hparams(hparams)

        edit_kwargs = {
            'subject': subjects,
            'prompts': questions,
            'target_new': targets,
            'summary_metrics': True,
            'keep_original_weight': True,
            'eval_model_id': args.model_eval,
        }

        metrics, edited_model, _ = editor.edit(**edit_kwargs)

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        with open(result_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f'\nModel: {model_id_format}, Editing dataset_sample with {editing_method} finished')
        del edited_model
        del editor
        gc.collect()
        torch.cuda.empty_cache()

    total_time = (time.time() - start_time) / 60
    print(f'\nOverall running time: {total_time:.2f} minutes')

# source hallu/bin/activate
# python3 edit_new_data.py --edit_method=ROME --data_size=2
# python3 edit_new_data.py --edit_method=ROME --model_name=gemma-12b --data_size=2
