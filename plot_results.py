import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import os
import argparse
import csv

base_path = "mia-result"

label_dict = {
    "WhiteMatter": 1,
    "GreyMatter": 2,
    "Hippocampus": 3,
    "Amygdala": 4,
    "Thalamus": 5,
}


def load_csv_grouped_by_pp(csv_file, metric="DICE"):
    """
    Load a CSV inside mia-result/, grouping values into pre and post
    based on 'PP' in the subject name.

    Expected CSV columns:
        seed, subject, label, metric, value
    """
    csv_path = os.path.join(base_path, csv_file)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    pre_data = {idx: [] for idx in label_dict.values()}
    post_data = {idx: [] for idx in label_dict.values()}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if "metric" not in row or "label" not in row or "value" not in row:
                raise ValueError("CSV missing label/metric/value columns")

            if row["metric"] != metric:
                continue

            label_name = row["label"]
            if label_name not in label_dict:
                continue

            try:
                val = float(row["value"])
            except Exception:
                continue

            idx = label_dict[label_name]
            subject = row.get("subject", "")

            if "PP" in subject:
                post_data[idx].append(val)
            else:
                pre_data[idx].append(val)

    return pre_data, post_data


def merge_pp(pre_data, post_data, pp_mode):
    """
    Merge or select pre/post values depending on pp_mode.

    pp_mode:
      - "pre"  -> only pre_data
      - "post" -> only post_data
      - "both" -> pre + post combined
    """
    merged = {idx: [] for idx in label_dict.values()}
    for idx in merged:
        if pp_mode == "pre":
            merged[idx] = pre_data[idx]
        elif pp_mode == "post":
            merged[idx] = post_data[idx]
        else:  # "both"
            merged[idx] = pre_data[idx] + post_data[idx]
    return merged


def plot_multiple_models(csv_list, metric="DICE", pp_mode="both"):
    n_models = len(csv_list)

    # ---------- CASE 1: only before or only after ----------
    if pp_mode in ("pre", "post"):
        datasets = []
        for csv_file in csv_list:
            pre, post = load_csv_grouped_by_pp(csv_file, metric)
            
            if pp_mode == "pre":
                datasets.append(pre)
            else:
                datasets.append(post)

        labels_present = [
            idx for idx in label_dict.values()
            if any(dataset[idx] for dataset in datasets)
        ]
        if not labels_present:
            print("No labels with data found.")
            return

        label_names = [name for name, idx in label_dict.items() if idx in labels_present]
        x_centers = np.arange(1, len(label_names) + 1)

        box_width = 0.7 / n_models
        offsets = np.linspace(-0.35 + box_width / 2, 0.35 - box_width / 2, n_models)

        plt.figure(figsize=(14, 6))
        cmap = get_cmap("viridis")
        colors = cmap(np.linspace(0.15, 0.85, n_models))

        for model_index, (csv_file, dataset) in enumerate(zip(csv_list, datasets)):
            values = [dataset[idx] for idx in label_dict.values() if idx in labels_present]
            positions = x_centers + offsets[model_index]

            bplot = plt.boxplot(values, positions=positions, widths=box_width, patch_artist=True)

            for patch in bplot["boxes"]:
                patch.set_facecolor(colors[model_index])

            means = [np.mean(v) if v else np.nan for v in values]
            ymin, ymax = plt.ylim()
            offset_y = (ymax - ymin) * 0.03

            for x, mean_val in zip(positions, means):
                if not np.isnan(mean_val):
                    plt.text(
                        x,
                        mean_val + offset_y,
                        f"{mean_val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8),
                    )

        plt.xticks(x_centers, label_names, rotation=35)
        plt.xlabel("Labels")
        plt.ylabel(metric)

        title = f"{metric} comparison of {n_models} models"
        title += " – before post-processing" if pp_mode == "pre" else " – after post-processing"

        plt.title(title, fontsize=16, fontweight="bold")

        for i, csv_file in enumerate(csv_list):
            label_text = "before post-processing" if pp_mode == "pre" else "after post-processing"
            plt.plot([], [], color=colors[i], label=f"{csv_file} ({label_text})")

        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # ---------- CASE 2: BOTH → before and after as separate blocks ----------
    pre_list = []
    post_list = []
    for csv_file in csv_list:
        pre, post = load_csv_grouped_by_pp(csv_file, metric)
        pre_list.append(pre)
        post_list.append(post)

    labels_present = [
        idx
        for idx in label_dict.values()
        if any(pre[idx] or post[idx] for pre, post in zip(pre_list, post_list))
    ]
    if not labels_present:
        print("No labels with data found.")
        return

    label_names = [name for name, idx in label_dict.items() if idx in labels_present]
    x_centers = np.arange(1, len(label_names) + 1)

    total_groups = 2 * n_models
    box_width = 0.8 / total_groups
    offsets = np.linspace(-0.4 + box_width / 2, 0.4 - box_width / 2, total_groups)

    plt.figure(figsize=(14, 6))
    cmap = get_cmap("viridis")
    colors = cmap(np.linspace(0.15, 0.85, n_models))

    for model_index, csv_file in enumerate(csv_list):
        color = colors[model_index]

        group_before = model_index
        group_after = model_index + n_models

        pos_before = x_centers + offsets[group_before]
        pos_after = x_centers + offsets[group_after]

        values_before = [pre_list[model_index][idx] for idx in label_dict.values() if idx in labels_present]
        values_after = [post_list[model_index][idx] for idx in label_dict.values() if idx in labels_present]

        # before post-processing
        b_before = plt.boxplot(
            values_before, positions=pos_before, widths=box_width, patch_artist=True
        )
        for patch in b_before["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # after post-processing
        b_after = plt.boxplot(
            values_after, positions=pos_after, widths=box_width, patch_artist=True
        )
        for patch in b_after["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(1.0)

    plt.xticks(x_centers, label_names, rotation=35)
    plt.xlabel("Labels")
    plt.ylabel(metric)
    plt.title(f"{metric} before vs after post-processing for {n_models} models",
              fontsize=16, fontweight="bold")

    # Annotate means
    ymin, ymax = plt.ylim()
    offset_y = (ymax - ymin) * 0.03

    for model_index in range(n_models):
        color = colors[model_index]

        group_before = model_index
        group_after = model_index + n_models

        pos_before = x_centers + offsets[group_before]
        pos_after = x_centers + offsets[group_after]

        values_before = [pre_list[model_index][idx] for idx in label_dict.values() if idx in labels_present]
        values_after = [post_list[model_index][idx] for idx in label_dict.values() if idx in labels_present]

        means_before = [np.mean(v) if v else np.nan for v in values_before]
        means_after = [np.mean(v) if v else np.nan for v in values_after]

        for x, mean_val in zip(pos_before, means_before):
            if not np.isnan(mean_val):
                plt.text(
                    x,
                    mean_val + offset_y,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.85),
                )

        for x, mean_val in zip(pos_after, means_after):
            if not np.isnan(mean_val):
                plt.text(
                    x,
                    mean_val + offset_y,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.85),
                )

    for i, csv_file in enumerate(csv_list):
        plt.plot([], [], color=colors[i], alpha=0.5,
                 label=f"{csv_file} (before post-processing)")
        plt.plot([], [], color=colors[i], alpha=1.0,
                 label=f"{csv_file} (after post-processing)")

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot segmentation metrics for one or several CSV files."
    )

    parser.add_argument(
        "--csv", type=str, nargs="+", required=True,
        help="One or more CSV files inside mia-result/"
    )

    parser.add_argument(
        "--metric", type=str, default="DICE",
        help="Metric to plot (e.g. DICE, JACRD, HDRFDST)"
    )

    parser.add_argument(
        "--pp", type=str, choices=["both", "pre", "post"], default="both",
        help="PP mode based on 'PP' in subject: 'pre', 'post', or 'both' (pre+post side-by-side)"
    )

    args = parser.parse_args()

    plot_multiple_models(
        csv_list=args.csv,
        metric=args.metric,
        pp_mode=args.pp
    )
