import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
random.seed(42)

class ProbeHelper:

    @staticmethod
    def remove_nan(tokens, activations):
            
        for idx,(source,target,activation) in  enumerate(zip(tokens['source'], tokens['target'], activations)):
            new_source = []
            new_target = []
            new_activation = np.array([])
            for source_token, target_token, activation_vector in zip(source, target, activation):
                if target_token == 'nan':
                    continue
                else:
                    new_source.append(source_token)
                    new_target.append(target_token)
                    if len(new_activation) == 0:
                        new_activation = activation_vector.reshape(1, -1)
                    else:
                        new_activation = np.vstack((new_activation, activation_vector))
            tokens['source'][idx] = new_source
            tokens['target'][idx] = new_target
            activations[idx] = new_activation

        return tokens, activations
    
    @staticmethod
    def balance_dataset(train_activations, train_labels, test_activations, test_labels, mapping, max_count):

        label2idx, idx2label, src2idx, idx2src = mapping
        # random sample the dataset where max_count is the maximum number of samples

        # Combine activations and labels
        combined_train = list(zip(train_activations, train_labels))
        combined_test = list(zip(test_activations, test_labels))

        # Randomly sample max_count elements (without replacement)
        # combined_train = random.sample(combined_train, k=min(max_count['train'], len(combined_train)))
        # combined_test = random.sample(combined_test, k=min(max_count['test'], len(combined_test)))

        # Get random indices for sampling
        train_indices = random.sample(range(len(combined_train)), k=min(max_count['train'], len(combined_train)))
        test_indices = random.sample(range(len(combined_test)), k=min(max_count['test'], len(combined_test)))

        # Unzip back
        combined_train = [combined_train[i] for i in train_indices]
        combined_test = [combined_test[i] for i in test_indices]
        train_activations, train_labels = zip(*combined_train)
        test_activations, test_labels = zip(*combined_test)

        train_labels = [idx2label[idx] for idx in train_labels]
        test_labels = [idx2label[idx] for idx in test_labels]

        # remap labels to 0,1,2,...
        label_encoder = LabelEncoder()
        label_encoder.fit(train_labels)

        label2idx = {label: idx for idx, label in enumerate(label_encoder.classes_)}
        # add unknown labels from test set to label2idx
        for label in test_labels:
            if label not in label2idx:
                label2idx[label] = len(label2idx)
        idx2label = {idx: label for label, idx in label2idx.items()}
        mapping = (label2idx, idx2label, src2idx, idx2src)

        train_labels = [label2idx[label] for label in train_labels]
        test_labels = [label2idx[label] for label in test_labels]

        # Convert back to list if needed
        train_activations = np.array(train_activations)
        train_labels = np.array(train_labels)
        test_activations = np.array(test_activations)
        test_labels = np.array(test_labels)

        return train_activations, train_labels, test_activations, test_labels, mapping, train_indices, test_indices