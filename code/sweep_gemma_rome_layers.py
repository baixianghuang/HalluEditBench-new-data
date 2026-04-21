import argparse
import json
import subprocess
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="hparams/ROME/gemma-12b.yaml")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--data-size", type=int, default=10)
    parser.add_argument("--dataset-path", default="../data/dataset_ox_new.json")
    parser.add_argument("--model-eval", default="gpt-4o-mini")
    parser.add_argument("--results-root", default="../results/gemma_layer_sweep")
    return parser.parse_args()


def load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, payload):
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def summarize_result(path: Path):
    data = json.loads(path.read_text())
    pre = [item["pre"]["edit_acc"][0] for item in data if item.get("pre", {}).get("edit_acc")]
    post = [item["post"]["edit_acc"][0] for item in data if item.get("post", {}).get("edit_acc")]
    times = [item["time"] for item in data if "time" in item]
    changed_outputs = sum(
        1
        for item in data
        if item.get("pre", {}).get("edit_output") != item.get("post", {}).get("edit_output")
    )
    return {
        "n": len(data),
        "pre_mean": sum(pre) / len(pre) if pre else None,
        "post_mean": sum(post) / len(post) if post else None,
        "changed_outputs": changed_outputs,
        "avg_time_sec": sum(times) / len(times) if times else None,
        "result_path": str(path),
    }


def main():
    args = parse_args()
    code_dir = Path(__file__).resolve().parent
    base_config_path = (code_dir / args.base_config).resolve()
    results_root = (code_dir / args.results_root).resolve()
    temp_config_dir = (code_dir / "hparams" / "ROME" / "_sweeps").resolve()
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    base_config = load_yaml(base_config_path)
    summary = []

    for layer in args.layers:
        variant_name = f"gemma-12b-layer-{layer}"
        variant_config_path = temp_config_dir / f"{variant_name}.yaml"
        layer_results_dir = results_root / f"layer_{layer}"

        config = dict(base_config)
        config["layers"] = [layer]
        dump_yaml(variant_config_path, config)

        cmd = [
            "python3",
            "edit_new_data.py",
            "--edit_method=ROME",
            f"--model_name=_sweeps/{variant_name}",
            f"--data_size={args.data_size}",
            f"--dataset_path={args.dataset_path}",
            f"--results_dir={layer_results_dir}",
            "--overwrite_result",
            f"--model_eval={args.model_eval}",
        ]

        print(f"\nRunning layer {layer}: {' '.join(cmd)}\n", flush=True)
        subprocess.run(cmd, cwd=code_dir, check=True)

        result_path = layer_results_dir / f"gemma_3_12b_it_ROME_{args.data_size}.json"
        metrics = summarize_result(result_path)
        metrics["layer"] = layer
        summary.append(metrics)

    summary.sort(key=lambda item: (item["post_mean"], -item["avg_time_sec"]), reverse=True)
    summary_path = results_root / f"summary_data_size_{args.data_size}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nLayer sweep summary:\n")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
