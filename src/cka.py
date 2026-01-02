""" 
code based on: https://github.com/RistoAle97/centered-kernel-alignment/tree/main
"""

import torch
import numpy as np
from tqdm.auto import tqdm
from typing import List, Sequence, Literal
from transformers import AutoModel, AutoTokenizer

from .encoding.transformers_extractor import extract_sentence_representations, get_model_and_tokenizer
 
 
class CKA:

    def __init__(
            self, 
            model_1: str, 
            model_2: str, 
            device: str = None,
            random_weights=False,):
        
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model1, self.tokenizer1 = get_model_and_tokenizer(
            model_1, device=self.device, random_weights=random_weights
        )

        self.model2, self.tokenizer2 = get_model_and_tokenizer(
            model_2, device=self.device, random_weights=random_weights
        )

    @staticmethod
    def collect_activations(
        model: AutoModel,
        tokenizer: AutoTokenizer,
        sentences: List[str],
        device: str = "cpu",
        ignore_embeddings: bool = False,
        token_aggregation: Literal["first", "last", "mean"] = "last",
        dtype: Literal["float32", "float64"] = "float32",
        include_special_tokens: bool = False,
        sentence_pooling: Literal["cls", "mean"] = "mean",
    ):
        
        num_layers = model.config.num_hidden_layers + 1 
        reps: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]


        tokenization_counts = {}
        with tqdm(total=len(sentences), desc="Forward passes", dynamic_ncols=True, leave=False) as pbar:
            for sentence in sentences:
                hidden_states, _ = extract_sentence_representations(
                    sentence,
                    model,
                    tokenizer,
                    device=device,
                    include_embeddings=(not ignore_embeddings),
                    token_aggregation=token_aggregation,
                    dtype=dtype,
                    include_special_tokens=include_special_tokens,
                    tokenization_counts=tokenization_counts,
                    output_level='sentence',     
                    sentence_pooling=sentence_pooling,    
                )
                for layer_idx, h in enumerate(hidden_states):
                    reps[layer_idx].append(h)
                pbar.update(1)
        
        # stack to (n × d) per layer
        reps = [np.stack(layer_list, axis=0).squeeze(1) for layer_list in reps]
        return reps

    @staticmethod
    def gram_linear(x):
        """Compute Gram (kernel) matrix for a linear kernel.

        Args:
            x: A num_examples x num_features matrix of features.

        Returns:
            A num_examples x num_examples Gram matrix of examples.
        """
        return x.dot(x.T)

    @staticmethod
    def gram_rbf(x, threshold=1.0):
        """Compute Gram (kernel) matrix for an RBF kernel.

        Args:
            x: A num_examples x num_features matrix of features.
            threshold: Fraction of median Euclidean distance to use as RBF kernel
            bandwidth. (This is the heuristic we use in the paper. There are other
            possible ways to set the bandwidth; we didn't try them.)

        Returns:
            A num_examples x num_examples Gram matrix of examples.
        """
        dot_products = x.dot(x.T)
        sq_norms = np.diag(dot_products)
        sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
        sq_median_distance = np.median(sq_distances)
        return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))

    @staticmethod
    def center_gram(gram, unbiased=False):
        """Center a symmetric Gram matrix.

        This is equvialent to centering the (possibly infinite-dimensional) features
        induced by the kernel before computing the Gram matrix.

        Args:
            gram: A num_examples x num_examples symmetric matrix.
            unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
            estimate of HSIC. Note that this estimator may be negative.

        Returns:
            A symmetric matrix with centered columns and rows.
        """
        if not np.allclose(gram, gram.T):
            raise ValueError('Input must be a symmetric matrix.')
        gram = gram.copy()

        if unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).
            n = gram.shape[0]
            np.fill_diagonal(gram, 0)
            means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            gram -= means[:, None]
            gram -= means[None, :]
            np.fill_diagonal(gram, 0)
        else:
            means = np.mean(gram, 0, dtype=np.float64)
            means -= np.mean(means) / 2
            gram -= means[:, None]
            gram -= means[None, :]

        return gram

    def cka_batch(self, x, y, debiased=False):
        """Compute CKA.

        Args:
            gram_x: A num_examples x num_examples Gram matrix.
            gram_y: A num_examples x num_examples Gram matrix.
            debiased: Use unbiased estimator of HSIC. CKA may still be biased.

        Returns:
            The value of CKA between X and Y.
        """

        if not isinstance(x, np.ndarray):
            x = x.numpy()
        if not isinstance(y, np.ndarray):
            y = y.numpy()

        gram_x = self.center_gram(self.gram_linear(x), unbiased=debiased)
        gram_y = self.center_gram(self.gram_linear(y), unbiased=debiased)

        # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
        # n*(n-3) (unbiased variant), but this cancels for CKA.
        scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

        normalization_x = np.linalg.norm(gram_x)
        normalization_y = np.linalg.norm(gram_y)
        cka_value = scaled_hsic / (normalization_x * normalization_y)
        
        if np.isnan(cka_value):
            return np.float64(0.0)  # Return 0 if CKA is NaN
        else:
            return cka_value


    def compute_cka(
        self,
        input_corpus_path: str,
        sentence_pooling = "mean",
        unbiased: bool = True
    ) -> List[torch.Tensor]:
        """Compute CKA between the activations of two models on a set of texts."""

        # Load texts from the input corpus
        with open(input_corpus_path, 'r') as file:
            texts = [line.strip() for line in file if line.strip()]  
        
        # Collect activations for both models
        acts1 = self.collect_activations(self.model1, self.tokenizer1, texts, self.device, sentence_pooling=sentence_pooling)
        acts2 = self.collect_activations(self.model2, self.tokenizer2, texts, self.device, sentence_pooling=sentence_pooling)

        n1, n2 = len(acts1), len(acts2)

        # Pairwise CKA matrix -------------------------------------------
        print("\nComputing pair‑wise CKA …")
        cka_mat = np.zeros((n1, n2), dtype=np.float64)
        for i, x in enumerate(acts1):
            for j, y in enumerate(acts2):
                cka_mat[i, j] = self.cka_batch(x, y, unbiased)

        return cka_mat

    def compute_cka_cross_corpus(
        self,
        corpus_path_1: str,
        corpus_path_2: str,
        sentence_pooling = "mean",
        unbiased: bool = True
    ) -> np.ndarray:
        """Compute CKA between the activations of two models on different corpora.
        
        Args:
            corpus_path_1: Path to corpus file for model 1
            corpus_path_2: Path to corpus file for model 2
            sentence_pooling: Pooling strategy for sentence representations
            unbiased: Whether to use unbiased estimator
            
        Returns:
            CKA matrix (n_layers_1 x n_layers_2)
        """
        # Load texts from both corpora
        with open(corpus_path_1, 'r') as file:
            texts1 = [line.strip() for line in file if line.strip()]
        
        with open(corpus_path_2, 'r') as file:
            texts2 = [line.strip() for line in file if line.strip()]
        
        # Ensure both corpora have the same number of sentences
        min_len = min(len(texts1), len(texts2))
        if len(texts1) != len(texts2):
            print(f"[warning] Corpora have different lengths ({len(texts1)} vs {len(texts2)}). Using first {min_len} sentences from each.")
            texts1 = texts1[:min_len]
            texts2 = texts2[:min_len]
        
        # Collect activations for both models on their respective corpora
        acts1 = self.collect_activations(self.model1, self.tokenizer1, texts1, self.device, sentence_pooling=sentence_pooling)
        acts2 = self.collect_activations(self.model2, self.tokenizer2, texts2, self.device, sentence_pooling=sentence_pooling)

        n1, n2 = len(acts1), len(acts2)

        # Pairwise CKA matrix -------------------------------------------
        print("\nComputing pair‑wise CKA (cross-corpus) …")
        cka_mat = np.zeros((n1, n2), dtype=np.float64)
        for i, x in enumerate(acts1):
            for j, y in enumerate(acts2):
                cka_mat[i, j] = self.cka_batch(x, y, unbiased)

        return cka_mat
