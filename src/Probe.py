from .encoding import transformers_extractor 
from .interpretation import create_tensors_tokens, create_tensors_sentences, linear_probe, ablation, summarize 
from .utils import load_data_token, load_data_sentence, load_activations, ProbeHelper
from collections import Counter
from typing import Literal
from typing import Optional, Dict, List
import os


class Probing:

    def __init__(
            self,
            probing_dialect_input_path_train,
            probing_dialect_label_path_train,
            probing_dialect_input_path_test,
            probing_dialect_label_path_test,
            activations_dir,
            probing_model,
            device,
            generate_activation,
            dataset_type: Literal['token', 'sentence'],
            save_activations=True,
            min_count=None,
        ):

        self.probing_dialect_input_path_train = os.path.normpath(probing_dialect_input_path_train)
        self.probing_dialect_label_path_train = os.path.normpath(probing_dialect_label_path_train)
        self.probing_dialect_input_path_test = os.path.normpath(probing_dialect_input_path_test)
        self.probing_dialect_label_path_test = os.path.normpath(probing_dialect_label_path_test)
        self.probing_dialect_and_probing_model_activation_path_train = os.path.normpath(activations_dir + '/activations_' + probing_model.split("/")[-1] + '_train.json')
        self.probing_dialect_and_probing_model_activation_path_test = os.path.normpath(activations_dir + '/activations_' + probing_model.split("/")[-1] + '_test.json')
        self.probing_model = probing_model
        self.dataset_type = dataset_type
        self.device = device

        if dataset_type not in ['token', 'sentence']:
            raise ValueError("dataset_type must be either 'token' or 'sentence'")
        
        
        
        if generate_activation:

            if not os.path.exists(os.path.dirname(self.probing_dialect_and_probing_model_activation_path_train)):
                os.makedirs(os.path.dirname(self.probing_dialect_and_probing_model_activation_path_train))
            
            if os.path.exists(self.probing_dialect_and_probing_model_activation_path_train):
                os.remove(self.probing_dialect_and_probing_model_activation_path_train)
            if os.path.exists(self.probing_dialect_and_probing_model_activation_path_test):
                os.remove(self.probing_dialect_and_probing_model_activation_path_test)

            transformers_extractor.extract_representations(
                model_desc=probing_model,
                input_corpus=self.probing_dialect_input_path_train,
                output_file=self.probing_dialect_and_probing_model_activation_path_train,
                device=device,
                token_aggregation="last",
                output_level=dataset_type
            )

            transformers_extractor.extract_representations(
                model_desc=probing_model,
                input_corpus=self.probing_dialect_input_path_test,
                output_file=self.probing_dialect_and_probing_model_activation_path_test,
                device=device,
                token_aggregation="last",
                output_level=dataset_type   
            )


        if dataset_type == 'token':

            self.probing_X_train, self.probing_y_train, self.probing_X_test, self.probing_y_test, self.probing_mapping, self.num_layers = self.prepare_data_token(
                
                paths = {
                    'train' : {
                        'input' : probing_dialect_input_path_train,
                        'label' : probing_dialect_label_path_train,
                        'activation' : self.probing_dialect_and_probing_model_activation_path_train
                    },
                    'test' : {
                        'input' : probing_dialect_input_path_test,
                        'label' : probing_dialect_label_path_test,
                        'activation' : self.probing_dialect_and_probing_model_activation_path_test
                    }
                }
            )


        elif dataset_type == 'sentence':

            self.probing_X_train, self.probing_y_train, self.probing_X_test, self.probing_y_test, self.probing_mapping, self.num_layers = self.prepare_data_sentence(
                paths = {
                    'train' : {
                        'input' : probing_dialect_input_path_train,
                        'label' : probing_dialect_label_path_train,
                        'activation' : self.probing_dialect_and_probing_model_activation_path_train
                    },
                    'test' : {
                        'input' : probing_dialect_input_path_test,
                        'label' : probing_dialect_label_path_test,
                        'activation' : self.probing_dialect_and_probing_model_activation_path_test
                    }
                }
            )  

        if min_count:
            self.probing_X_train, self.probing_y_train, self.probing_X_test, self.probing_y_test, self.probing_mapping, self.probing_train_indices, self.probing_test_indices = ProbeHelper.balance_dataset(self.probing_X_train, self.probing_y_train, self.probing_X_test, self.probing_y_test, self.probing_mapping, min_count)
            
        if generate_activation and not save_activations:

            if os.path.exists(self.probing_dialect_and_probing_model_activation_path_train):
                os.remove(self.probing_dialect_and_probing_model_activation_path_train)
            if os.path.exists(self.probing_dialect_and_probing_model_activation_path_test):
                os.remove(self.probing_dialect_and_probing_model_activation_path_test)
    
    @staticmethod
    def prepare_data_token(
            paths,
            mapping=None
        ):

        final_sentence_length = {}

        if paths['train']:

            final_sentence_length['train'] = {}

            input_path_train = paths['train']['input']
            label_path_train = paths['train']['label']
            activation_path_train = paths['train']['activation']
            
            activations_train, num_layers = load_activations(activation_path_train, 768)
            tokens_train = load_data_token(input_path_train, label_path_train, activations_train)
            tokens_train, activations_train = ProbeHelper.remove_nan(tokens_train, activations_train)
            
            with open(label_path_train) as labels_fp:
                line_tokens = []
                for line in labels_fp:
                    line_tokens += line.strip().split()
                c = Counter(line_tokens)
                c.pop('nan', None)
                del line_tokens
                task_specific_tag = max(c, key=c.get)

            if mapping:
                X_train, y_train, mapping = create_tensors_tokens(tokens_train, activations_train, task_specific_tag, mapping)
            else:
                X_train, y_train, mapping = create_tensors_tokens(tokens_train, activations_train, task_specific_tag)

        if paths['test']:

            input_path_test = paths['test']['input']
            label_path_test = paths['test']['label']
            activation_path_test = paths['test']['activation']

            if 'label_path_train' not in locals():
                with open(label_path_test) as labels_fp:
                    line_tokens = []
                    for line in labels_fp:
                        line_tokens += line.strip().split()
                    c = Counter(line_tokens)
                    c.pop('nan', None)
                    del line_tokens
                    task_specific_tag = max(c, key=c.get)

            activations_test, num_layers = load_activations(activation_path_test, 768)
            tokens_test = load_data_token(input_path_test, label_path_test, activations_test)
            tokens_test, activations_test = ProbeHelper.remove_nan(tokens_test, activations_test)

            if 'X_train' in locals() or mapping:
                X_test, y_test, mapping = create_tensors_tokens(tokens_test, activations_test, task_specific_tag, mapping)
            else:
                X_test, y_test, mapping = create_tensors_tokens(tokens_test, activations_test, task_specific_tag)

        if 'X_train' in locals() and 'X_test' not in locals():
            return X_train, y_train, None, None, mapping, num_layers
        elif 'X_train' not in locals() and 'X_test' in locals():
            return None, None, X_test, y_test, mapping, num_layers
        else:
            return X_train, y_train, X_test, y_test, mapping, num_layers

    @staticmethod
    def prepare_data_sentence(
            paths,
            mapping=None
        ):
        

        if paths['train']:

            input_path_train = paths['train']['input']
            label_path_train = paths['train']['label']
            activation_path_train = paths['train']['activation']
        
            activations_train, num_layers = load_activations(activation_path_train, 768)
            sentences_train = load_data_sentence(input_path_train, label_path_train)
            if mapping:
                X_train, y_train, mapping = create_tensors_sentences(sentences_train, activations_train, mappings=mapping)
            else:
                X_train, y_train, mapping = create_tensors_sentences(sentences_train, activations_train)

        if paths['test']:

            input_path_test = paths['test']['input']
            label_path_test = paths['test']['label']
            activation_path_test = paths['test']['activation']

            activations_test, num_layers = load_activations(activation_path_test, 768)
            sentences_test = load_data_sentence(input_path_test, label_path_test)

            if 'X_train' in locals() or mapping:
                X_test, y_test, mapping = create_tensors_sentences(sentences_test, activations_test, mapping)
            else:
                X_test, y_test, mapping = create_tensors_sentences(sentences_test, activations_test)


        if 'X_train' in locals() and 'X_test' not in locals():
            return X_train, y_train, None, None, mapping, num_layers
        elif 'X_train' not in locals() and 'X_test' in locals():
            return None, None, X_test, y_test, mapping, num_layers
        else:
            return X_train, y_train, X_test, y_test, mapping, num_layers

        
    def train_and_evaluate_probing_classifier(self, layer=None):

        label2idx, idx2label, src2idx, idx2src = self.probing_mapping


        if layer == None: 

            probe = linear_probe.train_logistic_regression_probe(self.probing_X_train, self.probing_y_train)

            #control task
            control_probe = linear_probe.train_logistic_regression_probe(self.probing_X_train, rng.permutation(self.probing_y_train))

            accuracy = linear_probe.evaluate_probe(probe, self.probing_X_test, self.probing_y_test, idx_to_class=idx2label, metric='accuracy')
            f1_score = linear_probe.evaluate_probe(probe, self.probing_X_test, self.probing_y_test, idx_to_class=idx2label, metric='f1', average='macro')

            control_probe_accuracy = linear_probe.evaluate_probe(control_probe, self.probing_X_test, self.probing_y_test, idx_to_class=idx2label, metric='accuracy')
            control_probe_f1_score = linear_probe.evaluate_probe(control_probe, self.probing_X_test, self.probing_y_test, idx_to_class=idx2label, metric='f1', average='macro')
        else:
            layer_X_train = ablation.filter_activations_by_layers(self.probing_X_train, [layer], self.num_layers)
            layer_X_test = ablation.filter_activations_by_layers(self.probing_X_test, [layer], self.num_layers)

            probe = linear_probe.train_logistic_regression_probe(layer_X_train, self.probing_y_train)

            #control task
            control_probe = linear_probe.train_logistic_regression_probe(layer_X_train, rng.permutation(self.probing_y_train))
            
            accuracy = linear_probe.evaluate_probe(probe, layer_X_test, self.probing_y_test, idx_to_class=idx2label, metric='accuracy')
            f1_score = linear_probe.evaluate_probe(probe, layer_X_test, self.probing_y_test, idx_to_class=idx2label, metric='f1', average='macro')

            control_probe_accuracy = linear_probe.evaluate_probe(control_probe, layer_X_test, self.probing_y_test, idx_to_class=idx2label, metric='accuracy')
            control_probe_f1_score = linear_probe.evaluate_probe(control_probe, layer_X_test, self.probing_y_test, idx_to_class=idx2label, metric='f1', average='macro')


        return {
            'probe' : probe,
            'accuracy': accuracy,
            'accuracy_selectivity' : accuracy['__OVERALL__'] - control_probe_accuracy['__OVERALL__'],
            'f1_score': f1_score,
            'f1_score_selectivity' : f1_score['__OVERALL__'] - control_probe_f1_score['__OVERALL__'],
            'layer' : layer
        }

    
