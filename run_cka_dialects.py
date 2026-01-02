import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from src import CKA


def safe_name(s: str) -> str:
    """Sanitize model ids and city names for filenames."""
    s = s.replace('/', '_').replace(' ', '_')
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def load_cities_models(json_path: str) -> Dict[str, Dict[str, List[str]]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run CKA between an MSA model and dialectal models across cities.")
    parser.add_argument("--msa_model", default="CAMeL-Lab/bert-base-arabic-camelbert-msa", help="HF model id for the MSA reference model")
    parser.add_argument("--json", default="models_dialect.json", help="Path to the cities→models JSON file")
    parser.add_argument("--corpus_dir", default="data/CKA/MADAR_Corpus", help="Directory containing city text files, e.g., MADAR_Fes.txt")
    parser.add_argument("--dialect_dir", default="dialects.txt", help="Directory containing needed dialects")
    parser.add_argument("--out_dir", default="results_cka", help="Directory to write CKA outputs")
    parser.add_argument("--device", default=None, help="Device to run on (e.g., cuda, cpu). Defaults to auto")
    parser.add_argument("--unbiased", action="store_true", help="Use unbiased estimator for HSIC in CKA")
    parser.add_argument("--msa_corpus", default="data/CKA/MADAR_Corpus/MADAR_MSA.txt", help="Path to MSA corpus file")
    args = parser.parse_args()

    cities = load_cities_models(args.json)
    with open(args.dialect_dir, 'r', encoding='utf-8') as f:
        dialects = [line.strip() for line in f if line.strip()]
    # Filter cities to only those with dialects we have
    cities = {city: cfg for city, cfg in cities.items() if cfg['dialect'] in dialects}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooling = "cls"

    # Check if MSA corpus exists
    if not os.path.isfile(args.msa_corpus):
        print(f"[warning] MSA corpus not found: {args.msa_corpus}")
        print("[warning] Will only run scenario 1: model_msa(da) vs model_da(da)")
        msa_corpus_exists = False
    else:
        msa_corpus_exists = True

    # Aggregate results across all pairs and scenarios
    aggregated = {
        "msa_model": args.msa_model,
        "msa_corpus": args.msa_corpus,
        "unbiased": bool(args.unbiased),
        "results": [],  # list of { city, dialect_model, scenario, corpus_model1, corpus_model2, cka }
    }

    # We'll re-use the MSA model for all runs; instantiate once per dialectal model pair
    for city, cfg in cities.items():
        city_file = os.path.join(args.corpus_dir, f"MADAR_{city}.txt")
        if not os.path.isfile(city_file):
            print(f"[skip] Corpus not found for {city}: {city_file}")
            continue

        models = cfg.get("models", [])
        if not models:
            print(f"[skip] No models listed for {city}")
            continue

        for dialect_model in models:
            if dialect_model == args.msa_model:
                # Skip comparing model to itself
                continue

            print(f"\n=== City: {city} | Dialect model: {dialect_model} ===")

            cka = CKA(args.msa_model, dialect_model, device=args.device)
            
            # Scenario 1: model_msa(da_sentence) vs model_da(da_sentence)
            print(f"  [Scenario 1] Both models on dialectal corpus: {city_file}")
            try:
                cka_mat_1 = cka.compute_cka(city_file, unbiased=args.unbiased, sentence_pooling=pooling)
                aggregated["results"].append({
                    "city": city,
                    "dialect_model": dialect_model,
                    "scenario": "msa_model(da) vs dialect_model(da)",
                    "corpus_model1": cfg['dialect'],
                    "corpus_model2": cfg['dialect'],
                    "cka": cka_mat_1.tolist(),
                })
                print(f"  [Scenario 1] ✓ Complete")
            except Exception as e:
                print(f"  [Scenario 1] ✗ Failed: {e}")

            if not msa_corpus_exists:
                continue

            # Scenario 2: model_msa(msa_sentence) vs model_da(msa_sentence)
            print(f"  [Scenario 2] Both models on MSA corpus: {args.msa_corpus}")
            try:
                cka_mat_2 = cka.compute_cka(args.msa_corpus, unbiased=args.unbiased, sentence_pooling=pooling)
                aggregated["results"].append({
                    "city": city,
                    "dialect_model": dialect_model,
                    "scenario": "msa_model(msa) vs dialect_model(msa)",
                    "corpus_model1": 'MSA',
                    "corpus_model2": 'MSA',
                    "cka": cka_mat_2.tolist(),
                })
                print(f"  [Scenario 2] ✓ Complete")
            except Exception as e:
                print(f"  [Scenario 2] ✗ Failed: {e}")

            # Scenario 3: model_msa(msa_sentence) vs model_da(da_sentence)
            print(f"  [Scenario 3] Cross-corpus: MSA model on {args.msa_corpus}, Dialect model on {city_file}")
            try:
                cka_mat_3 = cka.compute_cka_cross_corpus(
                    args.msa_corpus, 
                    city_file, 
                    unbiased=args.unbiased, 
                    sentence_pooling=pooling
                )
                aggregated["results"].append({
                    "city": city,
                    "dialect_model": dialect_model,
                    "scenario": "msa_model(msa) vs dialect_model(da)",
                    "corpus_model1": 'MSA',
                    "corpus_model2": cfg['dialect'],
                    "cka": cka_mat_3.tolist(),
                })
                print(f"  [Scenario 3] ✓ Complete")
            except Exception as e:
                print(f"  [Scenario 3] ✗ Failed: {e}")

    # Save aggregated JSON once at the end
    json_out = out_dir / "cka_results_all_scenarios_cls.json"
    import json as _json
    with open(json_out, "w", encoding="utf-8") as f:
        _json.dump(aggregated, f, ensure_ascii=False, indent=4)
    print(f"\n[summary saved] {json_out}")
    print(f"Total comparisons: {len(aggregated['results'])}")


if __name__ == "__main__":
    main()
