#!/usr/bin/env python3
"""
Cross-lingual Transfer Learning for Arabic Dialects - CKA Script
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np
import torch

from src import CKA

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CKA experiments for Arabic dialect models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        '--models-file', 
        type=str, 
        default='models_dialect.json',
        help='Path to JSON file containing cities and model configurations'
    )
    
    parser.add_argument(
        '--msa-model', 
        type=str,
        default='CAMeL-Lab/bert-base-arabic-camelbert-msa',
        help='HuggingFace model ID for the MSA reference model'
    )
    
    parser.add_argument(
        '--dialect-model', 
        type=str,
        help='Specific dialect model to compare with MSA model (if not specified, runs all models from models file)'
    )
    
    # Data configuration
    parser.add_argument(
        '--corpus-dir', 
        type=str, 
        default='data_cleaned/CKA/MADAR_Corpus',
        help='Directory containing city text files (e.g., MADAR_Fes.txt)'
    )
    
    parser.add_argument(
        '--msa-corpus', 
        type=str, 
        default='data_cleaned/CKA/MADAR_Corpus/MADAR_MSA.txt',
        help='Path to MSA corpus file'
    )
    
    parser.add_argument(
        '--dialects-file', 
        type=str, 
        default='dialects.txt',
        help='File containing list of dialects to process'
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='results_cka',
        help='Directory to save CKA results'
    )
    
    # Experiment configuration
    parser.add_argument(
        '--city', 
        type=str,
        help='Specific city to run (if not specified, runs all cities)'
    )
    
    parser.add_argument(
        '--scenario', 
        type=str,
        choices=['1', '2', '3', 'all'],
        default='all',
        help='Specific scenario to run: 1=both_on_dialect, 2=both_on_msa, 3=cross_corpus'
    )
    
    parser.add_argument(
        '--pooling', 
        type=str,
        default='mean',
        choices=['cls', 'mean'],
        help='Sentence pooling strategy'
    )
    
    parser.add_argument(
        '--unbiased', 
        action='store_true',
        help='Use unbiased estimator for HSIC in CKA computation'
    )
    
    # Hardware configuration
    parser.add_argument(
        '--device', 
        type=str, 
        choices=['cuda', 'cpu', 'auto', 'mps'],
        default='mps',
        help='Device to use for computation'
    )
    
    # Debugging and logging
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Print what would be done without actually running experiments'
    )
    
    parser.add_argument(
        '--skip-existing', 
        action='store_true',
        help='Skip experiments that already have results'
    )
    
    return parser.parse_args()


def get_device(device_arg):
    """Get the appropriate device based on argument."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    elif device_arg == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        return 'cpu'
    elif device_arg == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("Warning: MPS requested but not available, falling back to CPU")
        return 'cpu'
    return device_arg


def safe_name(s: str) -> str:
    """Sanitize model ids and city names for filenames."""
    s = s.replace('/', '_').replace(' ', '_')
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def load_cities_models(json_path: str, specific_city: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
    """Load city and model configurations from JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Models file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        cities = json.load(f)
    
    if specific_city:
        if specific_city not in cities:
            raise ValueError(f"City '{specific_city}' not found in {json_path}")
        return {specific_city: cities[specific_city]}
    
    return cities


def load_dialects(dialects_file: str) -> List[str]:
    """Load list of dialects from file."""
    if not os.path.exists(dialects_file):
        raise FileNotFoundError(f"Dialects file not found: {dialects_file}")
    
    with open(dialects_file, 'r', encoding='utf-8') as f:
        dialects = [line.strip() for line in f if line.strip()]
    
    return dialects


def filter_cities_by_dialects(cities: Dict, dialects: List[str]) -> Dict:
    """Filter cities to only those with dialects we have."""
    return {city: cfg for city, cfg in cities.items() if cfg['dialect'] in dialects}


def check_corpus_files(cities: Dict, corpus_dir: str, msa_corpus: str, verbose: bool = False) -> tuple:
    """Check which corpus files exist and filter cities accordingly."""
    msa_corpus_exists = os.path.isfile(msa_corpus)
    if not msa_corpus_exists and verbose:
        print(f"[warning] MSA corpus not found: {msa_corpus}")
        print("[warning] Will only run scenario 1: model_msa(da) vs model_da(da)")
    
    valid_cities = {}
    for city, cfg in cities.items():
        city_file = os.path.join(corpus_dir, f"MADAR_{city}.txt")
        if os.path.isfile(city_file):
            valid_cities[city] = cfg
        elif verbose:
            print(f"[skip] Corpus not found for {city}: {city_file}")
    
    return valid_cities, msa_corpus_exists


def get_results_path(results_dir: str, pooling: str) -> Path:
    """Get the path for results file."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / f"cka_results_all_scenarios_{pooling}.json"


def load_existing_results(results_path: Path) -> Dict:
    """Load existing results if they exist."""
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "results": [],
    }


def save_results(results: Dict, results_path: Path, verbose: bool = False):
    """Save results to file."""
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    if verbose:
        print(f"[results saved] {results_path}")


def result_exists(existing_results: Dict, city: str, dialect_model: str, scenario: str) -> bool:
    """Check if a specific result already exists."""
    for result in existing_results.get("results", []):
        if (result.get("city") == city and 
            result.get("dialect_model") == dialect_model and 
            result.get("scenario") == scenario):
            return True
    return False


def run_cka_scenario_1(cka: CKA, city_file: str, unbiased: bool, pooling: str, verbose: bool = False) -> Optional[np.ndarray]:
    """Run CKA Scenario 1: Both models on dialectal corpus."""
    try:
        if verbose:
            print(f"  [Scenario 1] Both models on dialectal corpus: {city_file}")
        cka_mat = cka.compute_cka(city_file, unbiased=unbiased, sentence_pooling=pooling)
        if verbose:
            print(f"  [Scenario 1] ✓ Complete")
        return cka_mat
    except Exception as e:
        if verbose:
            print(f"  [Scenario 1] ✗ Failed: {e}")
        return None


def run_cka_scenario_2(cka: CKA, msa_corpus: str, unbiased: bool, pooling: str, verbose: bool = False) -> Optional[np.ndarray]:
    """Run CKA Scenario 2: Both models on MSA corpus."""
    try:
        if verbose:
            print(f"  [Scenario 2] Both models on MSA corpus: {msa_corpus}")
        cka_mat = cka.compute_cka(msa_corpus, unbiased=unbiased, sentence_pooling=pooling)
        if verbose:
            print(f"  [Scenario 2] ✓ Complete")
        return cka_mat
    except Exception as e:
        if verbose:
            print(f"  [Scenario 2] ✗ Failed: {e}")
        return None


def run_cka_scenario_3(cka: CKA, msa_corpus: str, city_file: str, unbiased: bool, pooling: str, verbose: bool = False) -> Optional[np.ndarray]:
    """Run CKA Scenario 3: Cross-corpus comparison."""
    try:
        if verbose:
            print(f"  [Scenario 3] Cross-corpus: MSA model on {msa_corpus}, Dialect model on {city_file}")
        cka_mat = cka.compute_cka_cross_corpus(
            msa_corpus, 
            city_file, 
            unbiased=unbiased, 
            sentence_pooling=pooling
        )
        if verbose:
            print(f"  [Scenario 3] ✓ Complete")
        return cka_mat
    except Exception as e:
        if verbose:
            print(f"  [Scenario 3] ✗ Failed: {e}")
        return None


def should_run_scenario(scenario_arg: str, scenario_num: str) -> bool:
    """Check if a specific scenario should be run based on arguments."""
    return scenario_arg == 'all' or scenario_arg == scenario_num


def run_cka_experiment(msa_model: str, dialect_model: str, city: str, cfg: Dict, 
                      corpus_dir: str, msa_corpus: str, msa_corpus_exists: bool,
                      device: str, unbiased: bool, pooling: str, scenario: str,
                      verbose: bool = False) -> List[Dict]:
    """Run CKA experiment for a specific model pair and city."""
    city_file = os.path.join(corpus_dir, f"MADAR_{city}.txt")
    results = []
    
    if verbose:
        print(f"\n=== City: {city} | Dialect model: {dialect_model} ===")
    
    if dialect_model == msa_model:
        if verbose:
            print(f"[skip] Same model comparison: {dialect_model}")
        return results
    
    try:
        cka = CKA(msa_model, dialect_model, device=device)
        
        # Scenario 1: model_msa(da_sentence) vs model_da(da_sentence)
        if should_run_scenario(scenario, '1'):
            cka_mat_1 = run_cka_scenario_1(cka, city_file, unbiased, pooling, verbose)
            if cka_mat_1 is not None:
                results.append({
                    "city": city,
                    "dialect_model": dialect_model,
                    "scenario": "msa_model(da) vs dialect_model(da)",
                    "corpus_model1": cfg['dialect'],
                    "corpus_model2": cfg['dialect'],
                    "cka": cka_mat_1.tolist(),
                })
        
        # Scenario 2: model_msa(msa_sentence) vs model_da(msa_sentence)
        if msa_corpus_exists and should_run_scenario(scenario, '2'):
            cka_mat_2 = run_cka_scenario_2(cka, msa_corpus, unbiased, pooling, verbose)
            if cka_mat_2 is not None:
                results.append({
                    "city": city,
                    "dialect_model": dialect_model,
                    "scenario": "msa_model(msa) vs dialect_model(msa)",
                    "corpus_model1": 'MSA',
                    "corpus_model2": 'MSA',
                    "cka": cka_mat_2.tolist(),
                })
        
        # Scenario 3: model_msa(msa_sentence) vs model_da(da_sentence)
        if msa_corpus_exists and should_run_scenario(scenario, '3'):
            cka_mat_3 = run_cka_scenario_3(cka, msa_corpus, city_file, unbiased, pooling, verbose)
            if cka_mat_3 is not None:
                results.append({
                    "city": city,
                    "dialect_model": dialect_model,
                    "scenario": "msa_model(msa) vs dialect_model(da)",
                    "corpus_model1": 'MSA',
                    "corpus_model2": cfg['dialect'],
                    "cka": cka_mat_3.tolist(),
                })
    
    except Exception as e:
        if verbose:
            print(f"[error] Failed to create CKA instance: {e}")
    
    return results


def main():
    """Main function."""
    args = parse_arguments()
    
    if args.verbose:
        print(f"Arguments: {vars(args)}")
    
    # Setup
    device = get_device(args.device)
    
    if args.verbose:
        print(f"Device: {device}")
    
    # Load configurations
    try:
        cities = load_cities_models(args.models_file, args.city)
        dialects = load_dialects(args.dialects_file)
        cities = filter_cities_by_dialects(cities, dialects)
        cities, msa_corpus_exists = check_corpus_files(cities, args.corpus_dir, args.msa_corpus, args.verbose)
        
        if args.verbose:
            print(f"Cities to process: {list(cities.keys())}")
            print(f"MSA corpus available: {msa_corpus_exists}")
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading configurations: {e}")
        return 1
    
    if not cities:
        print("No valid cities found to process.")
        return 1
    
    # Setup results
    results_path = get_results_path(args.results_dir, args.pooling)
    aggregated = load_existing_results(results_path)
    
    # Initialize aggregated structure if needed
    if "msa_model" not in aggregated:
        aggregated.update({
            "msa_model": args.msa_model,
            "msa_corpus": args.msa_corpus,
            "unbiased": bool(args.unbiased),
            "pooling": args.pooling,
            "results": []
        })
    
    total_experiments = 0
    completed_experiments = 0
    
    # Process each city and its models
    for city, cfg in cities.items():
        models = cfg.get("models", [])
        
        if not models:
            if args.verbose:
                print(f"[skip] No models listed for {city}")
            continue
        
        # Filter models if specific dialect model is specified
        if args.dialect_model:
            if args.dialect_model in models:
                models = [args.dialect_model]
            else:
                if args.verbose:
                    print(f"[skip] Dialect model {args.dialect_model} not found for {city}")
                continue
        
        for dialect_model in models:
            if dialect_model == args.msa_model:
                continue
            
            total_experiments += 1
            
            # Check if we should skip existing results
            scenarios_to_check = ['1', '2', '3'] if args.scenario == 'all' else [args.scenario]
            scenario_names = {
                '1': "msa_model(da) vs dialect_model(da)",
                '2': "msa_model(msa) vs dialect_model(msa)",
                '3': "msa_model(msa) vs dialect_model(da)"
            }
            
            if args.skip_existing:
                skip_all = True
                for scenario_num in scenarios_to_check:
                    scenario_name = scenario_names[scenario_num]
                    if not result_exists(aggregated, city, dialect_model, scenario_name):
                        skip_all = False
                        break
                
                if skip_all:
                    if args.verbose:
                        print(f"[skip] All results exist for {city} - {dialect_model}")
                    completed_experiments += 1
                    continue
            
            if args.dry_run:
                print(f"[DRY RUN] Would process {city} with model {dialect_model}")
                continue
            
            # Run the experiment
            experiment_results = run_cka_experiment(
                args.msa_model, dialect_model, city, cfg,
                args.corpus_dir, args.msa_corpus, msa_corpus_exists,
                device, args.unbiased, args.pooling, args.scenario, args.verbose
            )
            
            if experiment_results:
                aggregated["results"].extend(experiment_results)
                completed_experiments += 1
                
                # Save results after each experiment
                save_results(aggregated, results_path, args.verbose)
            else:
                if args.verbose:
                    print(f"[error] Failed to get results for {city} - {dialect_model}")
    
    # Final summary
    if not args.dry_run:
        save_results(aggregated, results_path, args.verbose)
        print(f"\n[summary] {results_path}")
        print(f"Total comparisons: {len(aggregated['results'])}")
        print(f"Experiments completed: {completed_experiments}/{total_experiments}")
    else:
        print(f"\n[DRY RUN] Would run {total_experiments} experiments")
    
    return 0


if __name__ == "__main__":
    exit(main())