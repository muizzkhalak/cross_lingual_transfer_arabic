from src import Probing
import torch
import json
import os
import glob
import warnings
import pandas as pd
warnings.filterwarnings("ignore")


def main():

    balanced = False
    skip_irrelevant_model_dialect = False
    
    with open('models.json', 'r') as f:
        model_file = json.load(f)
        models = list(model_file.keys())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for model in models:

        model_dialect = model_file[model]['dialect']

        if os.path.exists(f'results_dialects{"_balanced" if balanced else ""}/results_{model.split('/')[-1].replace('-','_')}.json'):
            with open(f'results_dialects{"_balanced" if balanced else ""}/results_{model.split('/')[-1].replace('-','_')}.json', 'r') as f:
                results = json.load(f)
        else:
            results = {model: {}}

        for task in os.listdir('data_dialects/Probing'):

            if not results[model].get(task):
                results[model][task] = {}

            if balanced:
                min_count = {'train' : [], 'test' : []}
                for dialect in os.listdir(f'data_dialects/Probing/{task}/'):
                    for dataset in os.listdir(f'data_dialects/Probing/{task}/{dialect}'):
                        for file in os.listdir(f'data_dialects/Probing/{task}/{dialect}/{dataset}'):
                            if 'label' in file:
                                df = pd.read_csv(f'data_dialects/Probing/{task}/{dialect}/{dataset}/{file}', header=None, names=['label'])
                                df['label'] = df['label'].astype(str)
                                df['label'] = df['label'].str.strip()
                                # count total labels
                                df['label_count'] = df['label'].str.split().apply(len)
                                # total labels
                                total_labels = df['label_count'].sum()
                                if 'train' in file:
                                    min_count['train'].append(total_labels)
                                elif 'test' in file:
                                    min_count['test'].append(total_labels)
                min_count = {'train': min(min_count['train']), 'test': min(min_count['test'])}
            else:
                min_count = None  

            for dialect in os.listdir(f'data_dialects/Probing/{task}'):

                if skip_irrelevant_model_dialect:
                    if model_dialect not in ["Mixed", "MSA", "DA", dialect]:
                        print(f"Skipping {dialect} for model {model} as it is trained on {model_dialect}")
                        continue

                if not results[model][task].get(dialect):
                    results[model][task][dialect] = {}

                for dataset in os.listdir(f'data_dialects/Probing/{task}/{dialect}'):

                    print(f"Processing {dataset} with model {model}")

                    if results[model][task][dialect].get(dataset):
                        print(f"Results already exist for {dataset} with model {model}, skipping...")
                        continue

                    try:
                        probe = Probing(
                            probing_dialect_input_path_train=glob.glob(f'data_dialects/Probing/{task}/{dialect}/{dataset}/*_sentence_train.txt')[0],
                            probing_dialect_label_path_train=glob.glob(f'data_dialects/Probing/{task}/{dialect}/{dataset}/*_label_train.txt')[0],
                            probing_dialect_input_path_test=glob.glob(f'data_dialects/Probing/{task}/{dialect}/{dataset}/*_sentence_test.txt')[0],
                            probing_dialect_label_path_test=glob.glob(f'data_dialects/Probing/{task}/{dialect}/{dataset}/*_label_test.txt')[0],
                            activations_dir=f'activations/{dataset}',
                            probing_model=model,
                            device=device,
                            generate_activation=True,
                            dataset_type='sentence' if task == 'Sentiment' else 'token',
                            save_activations=False,
                            min_count=min_count
                        )

                        results[model][task][dialect][dataset] = {}
                        for layer in range(probe.num_layers):
                            result = probe.train_and_evaluate_probing_classifier(layer=layer)
                            results[model][task][dialect][dataset]['layer_' + str(layer)] = {"accuracy" : result["accuracy"], 
                                                                                             "f1_score" : result["f1_score"],
                                                                                             'accuracy_selectivity' : result['accuracy_selectivity'],
                                                                                             'f1_score_selectivity' : result['f1_score_selectivity']}
                        result = probe.train_and_evaluate_probing_classifier(layer=None)
                        results[model][task][dialect][dataset]['all_layers'] = {"accuracy" : result["accuracy"], 
                                                                                "f1_score" : result["f1_score"],
                                                                                'accuracy_selectivity' : result['accuracy_selectivity'],
                                                                                'f1_score_selectivity' : result['f1_score_selectivity']}

                    except Exception as e:
                        print(f"Error processing {dataset} with model {model}: {e}")
                        continue

                    print(results[model][task][dialect])

                    os.makedirs(f'results_dialects{"_balanced" if balanced else ""}', exist_ok=True)
                    with open(f'results_dialects{"_balanced" if balanced else ""}/results_{model.split('/')[-1].replace('-','_')}.json', 'w') as f:
                        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()