"""
Script to save non-normalized scores (e.g., max F1 across layers) for each (model, task, dialect, dataset) combo.
These scores correspond to Figure 5 in the paper.
"""
import os
import json
import glob
import pandas as pd
import argparse


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run probing experiments for Arabic dialect models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--results-dir', 
        type=str,
        help='Directory to load results from'
    )

    return parser.parse_args()


def load_overall_records(results_dir: str) -> pd.DataFrame:
    """Scan results_dialects/*.json and collect __OVERALL__ accuracy/f1 per layer.

    Columns:
      - source_file: JSON filename (traceability)
      - model: HF model id from file's top-level key
      - task: e.g., 'NER'
      - dialect: e.g., 'Egypt', 'MSA', ...
      - dataset: dataset name under the dialect
      - layer: 'layer_k' or 'all_layers'
      - accuracy_overall, f1_overall: scalar metrics if present
    """
    records = []
    json_files = glob.glob(os.path.join(results_dir, "*.json"))

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            # Normalize non-JSON tokens (NaN/Infinity) to valid JSON nulls
            text = (
                text.replace("NaN", "null")
                .replace("Infinity", "null")
                .replace("-Infinity", "null")
            )
            data = json.loads(text)
        except Exception as e:
            print(f"Skipping {os.path.basename(path)} due to parse error: {e}")
            continue

        source_file = os.path.basename(path)
        if not isinstance(data, dict):
            continue

        for model, task_dict in data.items():
            if not isinstance(task_dict, dict):
                continue
            for task, dialect_dict in task_dict.items():
                if not isinstance(dialect_dict, dict):
                    continue
                for dialect, dataset_dict in dialect_dict.items():
                    if not isinstance(dataset_dict, dict):
                        continue
                    for dataset, layers_dict in dataset_dict.items():
                        if not isinstance(layers_dict, dict):
                            continue
                        for layer, metrics in layers_dict.items():
                            if not isinstance(metrics, dict):
                                continue

                            acc = None
                            f1 = None
                            acc_dict = metrics.get("accuracy")
                            f1_dict = metrics.get("f1_score")
                            if isinstance(acc_dict, dict):
                                acc = acc_dict.get("__OVERALL__")
                            if isinstance(f1_dict, dict):
                                f1 = f1_dict.get("__OVERALL__")

                            if 'accuracy_selectivity' in metrics:
                                acc_selectivity = metrics.get('accuracy_selectivity')
                            else:
                                acc_selectivity = None

                            if 'f1_score_selectivity' in metrics:
                                f1_score_selectivity = metrics.get('f1_score_selectivity')
                            else:
                                f1_score_selectivity = None

                            if acc is not None or f1 is not None:
                                records.append(
                                    {
                                        "model": model,
                                        "task": task,
                                        "dialect": dialect,
                                        "dataset": dataset,
                                        "layer": layer,
                                        "accuracy_overall": acc,
                                        "f1_overall": f1,
                                        "accuracy_selectivity": acc_selectivity,
                                        "f1_score_selectivity": f1_score_selectivity,
                                    }
                                )

    return pd.DataFrame.from_records(records)


def max_f1_table_all_models_non_normalized(
    data: pd.DataFrame,
    tasks=("POS", "NER", "Sentiment"),
    f1_col="f1_overall",
    exclude_all_layers=True,
    model_mapping=None,
) -> pd.DataFrame:
    
    model_mapping = {
        'CAMeL-Lab/bert-base-arabic-camelbert-da': 'CAMeLBERT-DA',
        'CAMeL-Lab/bert-base-arabic-camelbert-mix': 'CAMeLBERT-MIX',
        'CAMeL-Lab/bert-base-arabic-camelbert-msa': 'CAMeLBERT-MSA',
        'SI2M-Lab/DarijaBERT': 'DarijaBERT',
        'alger-ia/dziribert': 'DziriBERT',
        'faisalq/EgyBERT': 'EgyBERT',
        'faisalq/SaudiBERT': 'SaudiBERT',
        'reemalyami/AraRoBERTa-DZ': 'AraRoBERTa-DZ',
        'reemalyami/AraRoBERTa-EGY': 'AraRoBERTa-EGY',
        'reemalyami/AraRoBERTa-JO': 'AraRoBERTa-JO',
        'reemalyami/AraRoBERTa-LB': 'AraRoBERTa-LB',
        'reemalyami/AraRoBERTa-OM': 'AraRoBERTa-OM',
        'reemalyami/AraRoBERTa-SA': 'AraRoBERTa-SA'
    }

    df = data.copy()

    # Filter tasks
    df = df[df["task"].isin(tasks)]

    # Optionally exclude the aggregated "all_layers" row
    if exclude_all_layers:
        df = df[df["layer"] != "all_layers"]

    # Optional nicer model names
    if model_mapping is not None:
        df["model"] = df["model"].replace(model_mapping)

    # 1) Max F1 per (task, dataset, dialect, model)
    grouped = (
        df.groupby(["task", "dataset", "dialect", "model"], as_index=False)[f1_col]
          .max()
    )

    # 2) Pivot: one column per model
    wide = (
        grouped.pivot(index=["task", "dataset", "dialect"], columns="model", values=f1_col)
               .reset_index()
    )

    # Rename index columns to match what you asked
    wide = wide.rename(columns={"task": "Task", "dataset": "Dataset", "dialect": "Dialect"})

    # NaN -> None
    wide = wide.where(pd.notnull(wide), None)

    # Optional: sort columns (keep first three fixed)
    fixed = ["Task", "Dataset", "Dialect"]
    model_cols = sorted([c for c in wide.columns if c not in fixed])
    wide = wide[fixed + model_cols]

    wide = wide[['Task', 'Dataset', 'Dialect', 
                 'CAMeLBERT-MSA', 'CAMeLBERT-DA','CAMeLBERT-MIX', 
                 'SaudiBERT', 'EgyBERT', 'DarijaBERT', 'DziriBERT',
                 'AraRoBERTa-SA', 'AraRoBERTa-EGY', 'AraRoBERTa-OM', 'AraRoBERTa-JO', 'AraRoBERTa-LB', 'AraRoBERTa-DZ']]

    return wide


def main():

    args = parse_arguments()

    if not args.results_dir:
        raise ValueError("Results directory must be specified with --results-dir")
    
    # Build the DataFrame
    results_dir = args.results_dir
    df_overall_balanced = load_overall_records(results_dir)

    # Order rows and preview
    if not df_overall_balanced.empty:
        def _layer_order(x: str) -> int:
            if x == "all_layers":
                return 10_000
            try:
                return int(str(x).split("_")[-1])
            except Exception:
                return 9_999

        df_overall_balanced["_order"] = df_overall_balanced["layer"].map(_layer_order)
        df_overall_balanced = (
            df_overall_balanced.sort_values(["model", "task", "dialect", "dataset", "_order"])\
                    .drop(columns=["_order"]).reset_index(drop=True)
        )

    print(
        f"Aggregated {len(df_overall_balanced)} rows from {len(glob.glob(os.path.join(results_dir, '*.json')))} files."
    )

    df_overall_balanced.head(5)
    df_overall_balanced = df_overall_balanced[df_overall_balanced['dataset'] != 'PADT_PUD']
    df_overall_balanced = df_overall_balanced[~((df_overall_balanced['dataset'] == 'CLEANANERcorp') & (df_overall_balanced['dialect'] != 'MSA'))]

    non_normalized = max_f1_table_all_models_non_normalized(df_overall_balanced, tasks=("POS","NER","Sentiment"))

    non_normalized.to_csv("results/non_normalized_best_layer_score_dialects.csv", index=False)

if __name__ == "__main__":
    main()