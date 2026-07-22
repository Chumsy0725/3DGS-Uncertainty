"""
Collect uncertainty_metrics.py results across all scenes under a dataset output folder
and print/save a summary table.

Usage:
    python collect_results.py -i <output_base>/<dataset_name> -s test
Expects the layout produced by run_dense.sh / run_sparse.sh:
    <output_base>/<dataset_name>/<scene>/renders/eval/<split>/uncertainty_metrics__<uncertainty_folder>.json
"""

import csv
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def load_scene_results(base_path: Path, split: str, uncertainty_folder: str, renders_folder: str):
    results = {}
    for scene_dir in sorted(p for p in base_path.iterdir() if p.is_dir()):
        metrics_path = scene_dir / renders_folder / "eval" / split / f"uncertainty_metrics__{uncertainty_folder}.json"
        if not metrics_path.is_file():
            continue
        with open(metrics_path) as f:
            results[scene_dir.name] = json.load(f)
    return results


def flatten_row(scene, info):
    q = info["results"]["quality_metrics"]
    l1 = info["results"]["L1"]["uncertainty_metrics"]
    dssim = info["results"]["DSSIM"]["uncertainty_metrics"]
    return {
        "scene": scene,
        "psnr": q["psnr"],
        "ssim": q["ssim"],
        "lpips": q["lpips"],
        "l1": q["l1"],
        "pearson_L1": l1["pearson"],
        "AUSE_L1": l1["AUSE"],
        "pearson_DSSIM": dssim["pearson"],
        "AUSE_DSSIM": dssim["AUSE"],
    }


def print_table(rows):
    columns = ["scene", "psnr", "ssim", "lpips", "l1", "pearson_L1", "AUSE_L1", "pearson_DSSIM", "AUSE_DSSIM"]
    widths = {c: max(len(c), *(len(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in columns}

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(fmt(r[c]).ljust(widths[c]) for c in columns))

    print("-" * len(header))
    mean_row = {"scene": "MEAN"}
    for c in columns[1:]:
        mean_row[c] = float(np.mean([r[c] for r in rows]))
    print(" | ".join(fmt(mean_row[c]).ljust(widths[c]) for c in columns))


if __name__ == "__main__":
    parser = ArgumentParser(description="Summarize uncertainty_metrics.py results across scenes")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the dataset output folder, e.g. output/mipnerf360")
    parser.add_argument("-s", "--split", type=str, default="test")
    parser.add_argument("--renders_folder", type=str, default="renders")
    parser.add_argument("-f", "--uncertainty_folder", type=str, default="error_masks")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path to write the summary CSV (default: <input>/summary_<split>.csv)")
    args = parser.parse_args()

    base_path = Path(args.input)
    assert base_path.is_dir(), f"{base_path} is not a directory"

    scene_results = load_scene_results(base_path, args.split, args.uncertainty_folder, args.renders_folder)
    if not scene_results:
        print(f"No results found under {base_path}/<scene>/{args.renders_folder}/eval/{args.split}/. "
              f"Run uncertainty_metrics.py for each scene first.")
        raise SystemExit(1)

    rows = [flatten_row(scene, info) for scene, info in scene_results.items()]

    print_table(rows)

    output_path = Path(args.output) if args.output else base_path / f"summary_{args.split}.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        mean_row = {"scene": "MEAN"}
        for c in rows[0].keys():
            if c == "scene":
                continue
            mean_row[c] = float(np.mean([r[c] for r in rows]))
        writer.writerow(mean_row)

    print(f"\nSaved summary to: {output_path}")
