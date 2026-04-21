import os
import json
import numpy as np
import pandas as pd


def get_avg_std(metric_list, percent=100, std_flag=False):
    mean_val = np.mean(metric_list)
    std_val = np.std(metric_list)
    if std_flag:
        return f"{mean_val*percent:.2f}±{std_val:.2f}"
    else:
        return np.round(mean_val*percent, 2)


def parse_filename(filename):
    """Parse filenames of the form {model}_{edit_method}_{num_samples}.json.

    The last segment is the sample count (numeric), the second-to-last is the
    edit method, and everything before that is the model name.
    """
    stem = filename.replace('.json', '')
    parts = stem.split('_')
    num_samples = parts[-1]
    edit_method = parts[-2]
    model = '_'.join(parts[:-2])
    return model, edit_method, num_samples


def summarize_results(results_folder, std_flag=False):
    rows = []

    for filename in sorted(os.listdir(results_folder)):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(results_folder, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        model, edit_method, num_samples = parse_filename(filename)

        efficacy_pre = get_avg_std([e['pre']['edit_acc'][0] for e in data], std_flag=std_flag)
        efficacy_post = get_avg_std([e['post']['edit_acc'][0] for e in data], std_flag=std_flag)
        row = {
            "model": model,
            "edit_method": edit_method,
            "n": num_samples,
            "efficacy_pre": efficacy_pre,
            "efficacy_post": efficacy_post,
            "hallucination_ratio_pre": 100 - efficacy_pre,
            "hallucination_ratio_post": 100 - efficacy_post,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(['model', 'edit_method', 'n']).reset_index(drop=True)
    return df


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    df = summarize_results(results_dir)
    print(df.to_string(index=False))
