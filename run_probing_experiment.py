#!/usr/bin/env python3
"""
Cross-lingual Transfer Learning for Arabic Dialects - Probing Script
"""

from src import Probing
import torch
import json
import os
import glob
import warnings
import pandas as pd
import argparse
import sys
warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run probing experiments for Arabic dialect models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        '--models-file', 
        type=str, 
        default='models.json',
        help='Path to JSON file containing model configurations'
    )
    
    parser.add_argument(
        '--model', 
        type=str,
        help='Specific model to run (if not specified, runs all models from models file)'
    )
    
    # Data configuration
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data_cleaned',
        help='Base directory containing probing data'
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='results_dialects',
        help='Directory to save results'
    )

    parser.add_argument(
        '--dialects-file', 
        type=str, 
        default='dialects.txt',
        help='File containing list of dialects to process'
    )
    
    # Experiment configuration
    parser.add_argument(
        '--experiment-type',
        type=str,
        choices=['MSA', 'Dialect'],
        help='Type of experiment - MSA overrides balanced and skip-irrelevant settings'
    )

    parser.add_argument(
        '--balanced', 
        action='store_true',
        help='Use balanced dataset (minimum count across dialects)'
    )
    
    parser.add_argument(
        '--skip-irrelevant', 
        action='store_true',
        help='Skip models that are not relevant for the dialect being tested'
    )
    
    parser.add_argument(
        '--task', 
        type=str,
        choices=['NER', 'POS', 'Sentiment'],
        help='Specific task to run (if not specified, runs all tasks)'
    )
    
    parser.add_argument(
        '--dialect', 
        type=str,
        help='Specific dialect to run (if not specified, runs all dialects)'
    )
    
    parser.add_argument(
        '--dataset', 
        type=str,
        help='Specific dataset to run (if not specified, runs all datasets)'
    )
    
    # Hardware configuration
    parser.add_argument(
        '--device', 
        type=str, 
        choices=['cuda', 'cpu', 'auto', 'mps'],
        default='cuda',
        help='Device to use for computation'
    )
    
    parser.add_argument(
        '--save-activations', 
        action='store_true',
        help='Save model activations to disk'
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
    
    return parser.parse_args()


def get_device(device_arg):
    """Get the appropriate device based on argument."""
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device_arg == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        return 'cpu'
    return device_arg


def load_models(models_file, specific_model=None):
    """Load model configurations."""
    if not os.path.exists(models_file):
        raise FileNotFoundError(f"Models file not found: {models_file}")
    
    with open(models_file, 'r') as f:
        model_file = json.load(f)
    
    if specific_model:
        if specific_model not in model_file:
            raise ValueError(f"Model '{specific_model}' not found in {models_file}")
        return {specific_model: model_file[specific_model]}
    
    return model_file


def get_min_count(task, data_dir, balanced):
    """Calculate minimum count for balanced datasets."""
    if not balanced:
        return None
    
    min_count = {'train': [], 'test': []}
    for dialect in os.listdir(f'{data_dir}/Probing/{task}/'):
        for dataset in os.listdir(f'{data_dir}/Probing/{task}/{dialect}'):
            for file in os.listdir(f'{data_dir}/Probing/{task}/{dialect}/{dataset}'):
                if 'label' in file:
                    df = pd.read_csv(f'{data_dir}/Probing/{task}/{dialect}/{dataset}/{file}', 
                                   header=None, names=['label'])
                    df['label'] = df['label'].astype(str)
                    df['label'] = df['label'].str.strip()
                    df['label_count'] = df['label'].str.split().apply(len)
                    total_labels = df['label_count'].sum()
                    if 'train' in file:
                        min_count['train'].append(total_labels)
                    elif 'test' in file:
                        min_count['test'].append(total_labels)
    
    return {'train': min(min_count['train']), 'test': min(min_count['test'])}


def should_skip_dialect(model_dialect, dialect, skip_irrelevant):
    """Check if dialect should be skipped for this model."""
    if not skip_irrelevant:
        return False
    return model_dialect not in ["Mixed", "MSA", "DA", dialect]


def get_results_path(results_dir, model, balanced):
    """Get the path for results file."""
    suffix = "_balanced" if balanced else ""
    model_name = model.split('/')[-1].replace('-', '_')
    return f'{results_dir}{suffix}/results_{model_name}.json'


def load_existing_results(results_path, model):
    """Load existing results if they exist."""
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {model: {}}


def save_results(results, results_path):
    """Save results to file."""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)


def load_dialects(dialects_file: str):
    """Load list of dialects from file."""
    if not os.path.exists(dialects_file):
        raise FileNotFoundError(f"Dialects file not found: {dialects_file}")
    
    with open(dialects_file, 'r', encoding='utf-8') as f:
        dialects = [line.strip() for line in f if line.strip()]
    
    return dialects


def run_probing_experiment(model, task, dialect, dataset, data_dir, device, 
                          save_activations, min_count, verbose):
    """Run a single probing experiment."""
    try:
        # Get file paths
        base_path = f'{data_dir}/Probing/{task}/{dialect}/{dataset}'
        train_sentence = glob.glob(f'{base_path}/*_sentence_train.txt')[0]
        train_label = glob.glob(f'{base_path}/*_label_train.txt')[0]
        test_sentence = glob.glob(f'{base_path}/*_sentence_test.txt')[0]
        test_label = glob.glob(f'{base_path}/*_label_test.txt')[0]
        
        if verbose:
            print(f"  Files found: {len(glob.glob(f'{base_path}/*'))} files")
        
        # Create probe
        probe = Probing(
            probing_dialect_input_path_train=train_sentence,
            probing_dialect_label_path_train=train_label,
            probing_dialect_input_path_test=test_sentence,
            probing_dialect_label_path_test=test_label,
            activations_dir=f'activations/{dataset}',
            probing_model=model,
            device=device,
            generate_activation=True,
            dataset_type='sentence' if task == 'Sentiment' else 'token',
            save_activations=save_activations,
            min_count=min_count
        )
        
        # Run experiments
        experiment_results = {}
        
        # Individual layers
        for layer in range(probe.num_layers):
            if verbose:
                print(f"    Processing layer {layer}/{probe.num_layers-1}")
            result = probe.train_and_evaluate_probing_classifier(layer=layer)
            experiment_results[f'layer_{layer}'] = {
                "accuracy": result["accuracy"],
                "f1_score": result["f1_score"],
                'accuracy_selectivity': result['accuracy_selectivity'],
                'f1_score_selectivity': result['f1_score_selectivity']
            }
        
        # All layers
        if verbose:
            print("    Processing all layers combined")
        result = probe.train_and_evaluate_probing_classifier(layer=None)
        experiment_results['all_layers'] = {
            "accuracy": result["accuracy"],
            "f1_score": result["f1_score"],
            'accuracy_selectivity': result['accuracy_selectivity'],
            'f1_score_selectivity': result['f1_score_selectivity']
        }
        
        return experiment_results
        
    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        return None


def main():
    """Main function."""
    args = parse_arguments()
    
    if args.verbose:
        print(f"Arguments: {vars(args)}")
    
    # Validation: if experiment_type is not specified, require specific dialect, task, and model
    if args.experiment_type is None:
        missing_args = []
        if args.dialect is None:
            missing_args.append('--dialect')
        if args.task is None:
            missing_args.append('--task')
        if args.model is None:
            missing_args.append('--model')
        
        if missing_args:
            raise ValueError(f"When --experiment-type is not specified, the following arguments are required: {', '.join(missing_args)}")
    
    # Override settings for MSA experiment type
    if args.experiment_type == 'MSA':
        args.balanced = False
        args.skip_irrelevant = False
        if args.verbose:
            print("MSA experiment type detected: setting balanced=False, skip_irrelevant=False")
    elif args.experiment_type == 'Dialect':
        args.balanced = True
        args.skip_irrelevant = True
        if args.verbose:
            print("Dialect experiment type detected: setting balanced=True, skip_irrelevant=True")
    
    # Setup
    device = get_device(args.device)
    model_file = load_models(args.models_file, args.model)
    
    if args.verbose:
        print(f"Device: {device}")
        print(f"Models to process: {list(model_file.keys())}")
    
    # Get tasks, dialects, and datasets to process
    if args.task:
        tasks = [args.task]
    else:
        tasks = [t for t in os.listdir(f'{args.data_dir}/Probing') if os.path.isdir(f'{args.data_dir}/Probing/{t}')]
    
    for model in model_file:
        model_dialect = model_file[model]['dialect']
        results_path = get_results_path(args.results_dir, model, args.balanced)
        results = load_existing_results(results_path, model)
        
        if args.verbose:
            print(f"\nProcessing model: {model} (dialect: {model_dialect})")
        
        for task in tasks:
            if args.verbose:
                print(f"  Task: {task}")
            
            if not results[model].get(task):
                results[model][task] = {}
            
            # Calculate min_count for balanced datasets
            min_count = get_min_count(task, args.data_dir, args.balanced)
            
            # Get dialects for this task
            if args.experiment_type == 'MSA':
                dialects = ['MSA']
            else:
                if args.dialect:
                    dialects = [args.dialect] if args.dialect in load_dialects(args.dialects_file) else []
                else:
                    dialects = load_dialects(args.dialects_file)
            
            for dialect in dialects:
                if should_skip_dialect(model_dialect, dialect, args.skip_irrelevant):
                    if args.verbose:
                        print(f"    Skipping {dialect} for model {model} (trained on {model_dialect})")
                    continue
                
                if args.verbose:
                    print(f"    Dialect: {dialect}")
                
                if not results[model][task].get(dialect):
                    results[model][task][dialect] = {}
                
                # Get datasets for this dialect
                if args.dataset:
                    datasets = [args.dataset] if args.dataset in os.listdir(f'{args.data_dir}/Probing/{task}/{dialect}') else []
                else:
                    datasets = os.listdir(f'{args.data_dir}/Probing/{task}/{dialect}')
                
                for dataset in datasets:
                    print(f"      Processing {dataset} with model {model}")
                    
                    if results[model][task][dialect].get(dataset):
                        print(f"      Results already exist for {dataset}, skipping...")
                        continue
                    
                    if args.dry_run:
                        print(f"      [DRY RUN] Would process {dataset}")
                        continue
                    
                    experiment_results = run_probing_experiment(
                        model, task, dialect, dataset, args.data_dir, device,
                        args.save_activations, min_count, args.verbose
                    )
                    
                    if experiment_results:
                        results[model][task][dialect][dataset] = experiment_results
                        
                        if args.verbose:
                            print(f"      Results: {results[model][task][dialect][dataset]['all_layers']}")
                        
                        # Save results after each dataset
                        save_results(results, results_path)
                    else:
                        print(f"      Failed to process {dataset}")
    
    print("\nProcessing complete!")
    if not args.dry_run:
        print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()