"""Representations Extractor for ``transformers`` toolkit models.

Module that given a file with input sentences and a ``transformers``
model, extracts representations from all layers of the model. The script
supports aggregation over sub-words created due to the tokenization of
the provided model.

Can also be invoked as a script as follows:
    ``python -m neurox.data.extraction.transformers_extractor``
"""

import argparse
import sys
import re

import numpy as np
import torch
from typing import Literal

from ..utils.writer import ActivationsWriter

from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
    """
    Automatically get the appropriate ``transformers`` model and tokenizer based
    on the model description

    Parameters
    ----------
    model_desc : str
        Model description; can either be a model name like ``bert-base-uncased``,
        a comma separated list indicating <model>,<tokenizer> (since 1.0.8),
        or a path to a trained model

    device : str, optional
        Device to load the model on, cpu or gpu. Default is cpu.

    random_weights : bool, optional
        Whether the weights of the model should be randomized. Useful for analyses
        where one needs an untrained model.

    Returns
    -------
    model : transformers model
        An instance of one of the transformers.modeling classes
    tokenizer : transformers tokenizer
        An instance of one of the transformers.tokenization classes
    """
    model_desc = model_desc.split(",")
    if len(model_desc) == 1:
        model_name = model_desc[0]
        tokenizer_name = model_desc[0]
    else:
        model_name = model_desc[0]
        tokenizer_name = model_desc[1]
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if random_weights:
        print("Randomizing weights")
        model.init_weights()

    return model, tokenizer


def aggregate_repr(state, start, end, token_aggregation):
    """
    Function that aggregates activations/embeddings over a span of subword tokens.
    This function will usually be called once per word. For example, if we had the sentence::

        This is an example

    which is tokenized by BPE into::

        this is an ex @@am @@ple

    The function should be called 4 times::

        aggregate_repr(state, 0, 0, token_aggregation)
        aggregate_repr(state, 1, 1, token_aggregation)
        aggregate_repr(state, 2, 2, token_aggregation)
        aggregate_repr(state, 3, 5, token_aggregation)

    Returns a zero vector if end is less than start, i.e. the request is to
    aggregate over an empty slice.

    Parameters
    ----------
    state : numpy.ndarray
        Matrix of size [ NUM_LAYERS x NUM_SUBWORD_TOKENS_IN_SENT x LAYER_DIM]
    start : int
        Index of the first subword of the word being processed
    end : int
        Index of the last subword of the word being processed
    token_aggregation : {'first', 'last', 'average'}
        Aggregation method for combining subword activations

    Returns
    -------
    word_vector : numpy.ndarray
        Matrix of size [NUM_LAYERS x LAYER_DIM]
    """
    if end < start:
        sys.stderr.write(
            "WARNING: An empty slice of tokens was encountered. "
            + "This probably implies a special unicode character or text "
            + "encoding issue in your original data that was dropped by the "
            + "transformer model's tokenizer.\n"
        )
        return np.zeros((state.shape[0], state.shape[2]))
    if token_aggregation == "first":
        return state[:, start, :]
    elif token_aggregation == "last":
        return state[:, end, :]
    elif token_aggregation == "average":
        return np.average(state[:, start : end + 1, :], axis=1)



# def extract_sentence_representations(
#     sentence: str,
#     model,
#     tokenizer,
#     device: str = "cpu",
#     include_embeddings: bool = True,
#     token_aggregation: str = "last",
#     dtype: str = "float32",
#     include_special_tokens: bool = False,
#     tokenization_counts: dict = None,
#     output_level: str = "sentence",       # "token"  (default)  or "sentence"
#     sentence_pooling: str = "mean",    # "mean" or "cls"   (only used if output_level == "sentence")
#     verbose: bool = False,             # replaces the former print-spree
# ):
#     """
#     Extract contextualised representations for either
#     - every token in `sentence`   (output_level='token'), **or**
#     - the whole sentence          (output_level='sentence').

#     Parameters
#     ----------
#     output_level : {'token', 'sentence'}
#         Determines what kind of vector(s) the function returns.
#     sentence_pooling : {'mean', 'cls'}
#         Which pooling strategy to apply when `output_level='sentence'`.
#         * 'mean' – average of last-layer hidden states (excl. special tokens)
#         * 'cls'  – hidden state at position 0 (BERT-style CLS token)

#     Returns
#     -------
#     np.ndarray
#         * Token level  : shape (NUM_LAYERS, NUM_TOKENS, HIDDEN_SIZE)
#         * Sentence lvl : shape (NUM_LAYERS, HIDDEN_SIZE)   ⟵ one vector per layer
#     list | None
#         * Token level  : detokenised words (plus specials if requested)
#         * Sentence lvl : None
#     """

#     if tokenization_counts is None:
#         tokenization_counts = {}

#     if output_level not in {"token", "sentence"}:
#         raise ValueError("`output_level` must be 'token' or 'sentence'")

#     if sentence_pooling not in {"mean", "cls"}:
#         raise ValueError("`sentence_pooling` must be 'mean' or 'cls'")

#     special_tokens = [t for t in tokenizer.all_special_tokens if t != tokenizer.unk_token]
#     special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

#     original_tokens = sentence.split(" ")

#     # ------------------------------------------------------------------ #
#     # 1) Build helper sentence with boundary markers to measure subwords #
#     # ------------------------------------------------------------------ #
#     boundary_token = 'a'
#     tmp_tokens = []
#     if original_tokens:
#         tmp_tokens.append(f"{original_tokens[0]} {boundary_token}")
#     tmp_tokens += [f"{boundary_token} {x} {boundary_token}" for x in original_tokens[1:-1]]
#     if len(original_tokens) > 1:
#         tmp_tokens.append(f"{boundary_token} {original_tokens[-1]}")
    

#     # Guard – length parity
#     assert len(tmp_tokens) == len(original_tokens)

#     # Check if tokenizer encodes space
#     if tokenizer.encode(" " + boundary_token)[1] != tokenizer.encode(boundary_token)[0]:
#         special_tokens_ids += tokenizer.encode(" ")[1:-1]  # exclude cls/sep

#     # ---------------------------------------------------- #
#     # 2) Count sub-words for each original token (cached)  #
#     # ---------------------------------------------------- #
    
#     with torch.no_grad():
#         token_lengths = 0
#         for i, token in enumerate(tmp_tokens):
#             token_ids = [
#                 x for x in tokenizer.encode(token) if x not in special_tokens_ids
#             ]
#             # Ignore the added letter tokens
#             if i != 0 and i != len(tmp_tokens) - 1:
#                 # Word appearing in the middle of the sentence
#                 token_ids = token_ids[1:-1]
#             elif i == 0:
#                 # Word appearing at the beginning
#                 token_ids = token_ids[:-1]
#             else:
#                 # Word appearing at the end
#                 token_ids = token_ids[1:]

#             token_lengths += len(token_ids)
#             if token_lengths > 512-2:
#                 tmp_tokens = tmp_tokens[:i]
#                 break

#             if token in tokenization_counts:
#                 if tokenization_counts[token] != len(token_ids):
#                     raise RuntimeError(
#                         f"Cache mismatch for token '{token}': "
#                         f"{tokenization_counts[token]} vs {len(token_ids)}"
#                     )
#             else:
#                 tokenization_counts[token] = len(token_ids)

            

#         # ----------------------------------------------- #
#         # 3) Forward pass through the model               #
#         # ----------------------------------------------- #
#         try:
#             ids = tokenizer.encode(sentence, truncation=True, max_length=512-2)
#             inputs = torch.tensor([ids], device=device)
#             hidden_states = model(inputs, output_hidden_states=True)[-1]  # tuple(len=L+1)
#         except:
#             pass

#         if include_embeddings:
#             layers = [h[0].cpu().numpy() for h in hidden_states]      # L+1 layers
#         else:
#             layers = [h[0].cpu().numpy() for h in hidden_states[1:]]  # exclude embed layer
#         layers = np.array(layers, dtype=dtype)                        # (L, S, H)

#     # -------------------------------------- #
#     # 4) TOKEN-LEVEL path (unchanged logic)  #
#     # -------------------------------------- #
#     if output_level == "token":
#         # -- identical detokenisation and aggregation block ------------
#         seg_tokens = tokenizer.convert_ids_to_tokens(ids)
#         idx_special = [i for i, t in enumerate(ids) if t in special_tokens_ids]

#         # Optionally drop specials
#         if not include_special_tokens:
#             keep_idx = [i for i in range(len(ids)) if i not in idx_special]
#             seg_tokens = [seg_tokens[i] for i in keep_idx]
#             layers = layers[:, keep_idx, :]
#             idx_special = []

#         # rebuild
#         final = np.zeros(
#             (
#                 layers.shape[0],
#                 len(original_tokens) + len(idx_special),
#                 layers.shape[2],
#             ),
#             dtype=dtype,
#         )
#         detok = []
#         ptr = 0
#         # deal with specials BEFORE first word
#         while idx_special and ptr == idx_special[0]:
#             detok.append(seg_tokens[ptr])
#             final[:, len(detok) - 1, :] = layers[:, ptr, :]
#             idx_special.pop(0)
#             ptr += 1

#         for tok_text in tmp_tokens:
#             sw_count = tokenization_counts[tok_text]
#             start, end = ptr, ptr + sw_count  # inclusive end later
#             if sw_count:
#                 if token_aggregation == "first":
#                     vec = layers[:, start, :]
#                 elif token_aggregation == "last":
#                     vec = layers[:, end - 1, :]
#                 else:  # "average"
#                     vec = layers[:, start:end, :].mean(axis=1)
#                 detok.append("".join(seg_tokens[start:end]))
#                 final[:, len(detok) - 1, :] = vec
#             ptr = end

#             # interleave specials that appear between words
#             while idx_special and ptr == idx_special[0]:
#                 detok.append(seg_tokens[ptr])
#                 final[:, len(detok) - 1, :] = layers[:, ptr, :]
#                 idx_special.pop(0)
#                 ptr += 1

#         if verbose:
#             print(f"Tokenised  ({len(seg_tokens):03d}): {seg_tokens}")
#             print(f"Detokenised({len(detok):03d}): {detok}")

#         return final, detok

#     # ------------------------------------------- #
#     # 5) SENTENCE-LEVEL path (new functionality!) #
#     # ------------------------------------------- #
#     else:  # output_level == 'sentence'
#         # Choose the layer(s) to pool – we keep *all* layers,
#         # giving one sentence vector per layer
#         if sentence_pooling == "cls":
#             sent_repr = layers[:, 0, :]           # (L, H)
#             sent_repr = np.expand_dims(sent_repr, axis=1)  # (L, 1, H)
#         else:  # 'mean'
#             # exclude special tokens for mean
#             non_special_idx = [i for i, t in enumerate(ids) if t not in special_tokens_ids]
#             sent_repr = layers[:, non_special_idx, :].mean(axis=1) # (L, H)
#             sent_repr = np.expand_dims(sent_repr, axis=1)  # (L, 1, H)  

#         if verbose:
#             print(f"Sentence pooling: {sentence_pooling.upper()}  -> shape {sent_repr.shape}")

#         return sent_repr.astype(dtype, copy=False), [sentence]
    

def extract_sentence_representations(
    sentence,
    model,
    tokenizer,
    device="cpu",
    include_embeddings=True,
    token_aggregation="last",
    dtype="float32",
    include_special_tokens=False,
    tokenization_counts={},
    output_level: Literal["token", "sentence"] = "token",
    sentence_pooling: Literal["cls", "mean"] = None
):
    """
    Get representations for a single sentence

    The extractor runs a detokenization procedure to combine subwords
    automatically. For instance, a sentence "Hello, how are you?" may be
    tokenized by the model as "Hell @@o , how are you @@?". This extractor
    automatically detokenizes the subtokens back into the original token.


    Parameters
    ----------
    sentence : str
        Sentence for which the extraction needs to be done. The returned output
        will have representations for exactly the same number of elements as
        tokens in this sentence (counted by `sentence.split(' ')`).

    model : transformers model
        An instance of one of the transformers.modeling classes

    tokenizer : transformers tokenizer
        An instance of one of the transformers.tokenization classes

    device : str, optional
        Specifies the device (CPU/GPU) on which the extraction should be
        performed. Defaults to 'cpu'

    include_embeddings : bool, optional
        Whether the embedding layer should be included in the final output, or
        just regular layers. Defaults to True

    token_aggregation : {'first', 'last', 'average'}, optional
        Aggregation method for combining subword activations. Defaults to 'last'

    dtype : str, optional
        Data type in which the activations will be stored. Supports all numpy
        based tensor types. Common values are 'float32' and 'float16'. Defaults
        to 'float16'

    include_special_tokens : bool, optional
        Whether or not to special tokens in the extracted representations.
        Special tokens are tokens not present in the original sentence, but are
        added by the tokenizer, such as [CLS], [SEP] etc.

    tokenization_counts : dict, optional
        Tokenization counts to use across a dataset for efficiency

    Returns
    -------
    final_hidden_states : numpy.ndarray
        Numpy Matrix of size [``NUM_LAYERs`` x ``NUM_TOKENS`` x ``NUM_NEURONS``].

    detokenizer : list
        List of detokenized words. This will have the same number of elements as
        tokens in the original sentence, plus special tokens if requested. Each element
        preserves tokenization artifacts (such as `##`, `@@` etc) to enable further
        automatic processing.
    """

    if output_level not in ['token', 'sentence']:
        raise ValueError("`output_level` must be 'token' or 'sentence'")
    
    if output_level == 'sentence' and sentence_pooling not in ['cls', 'mean']:
        raise ValueError("`sentence_pooling` must be 'cls' or 'mean' when `output_level` is 'sentence'")

    special_tokens = [
        x for x in tokenizer.all_special_tokens if x != tokenizer.unk_token
    ]
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    #sentence clean up
    sentence = re.sub(' +', ' ', sentence)
    
    original_tokens = sentence.split(" ")

    # Add letters and spaces around each word since some tokenizers are context sensitive
    tmp_tokens = []
    if len(original_tokens) > 0:
        tmp_tokens.append(f"{original_tokens[0]} a")
    tmp_tokens += [f"a {x} a" for x in original_tokens[1:-1]]
    if len(original_tokens) > 1:
        tmp_tokens.append(f"a {original_tokens[-1]}")

    assert len(original_tokens) == len(
        tmp_tokens
    ), f"Original: {original_tokens}, Temp: {tmp_tokens}"

    # Check if tokenizer encodes space
    if tokenizer.encode(" " + 'a')[1] != tokenizer.encode('a')[0]:
        special_tokens_ids += tokenizer.encode(" ")[1:-1]  # exclude cls/sep

    with torch.no_grad():
        # Get tokenization counts if not already available
        for token_idx, token in enumerate(tmp_tokens):
            tok_ids = [
                x for x in tokenizer.encode(token) if x not in special_tokens_ids
            ]
            # Ignore the added letter tokens
            if token_idx != 0 and token_idx != len(tmp_tokens) - 1:
                # Word appearing in the middle of the sentence
                tok_ids = tok_ids[1:-1]
            elif token_idx == 0:
                # Word appearing at the beginning
                tok_ids = tok_ids[:-1]
            else:
                # Word appearing at the end
                tok_ids = tok_ids[1:]

            if token in tokenization_counts:
                assert tokenization_counts[token] == len(
                    tok_ids
                ), "Got different tokenization for already processed word"
            else:
                tokenization_counts[token] = len(tok_ids)
        ids = tokenizer.encode(sentence, truncation=True, max_length=510)  # leave space for cls/sep
        input_ids = torch.tensor([ids]).to(device)
        # Hugging Face format: tuple of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        # Tuple has 13 elements for base model: embedding outputs + hidden states at each layer
        all_hidden_states = model(input_ids)[-1]

        if include_embeddings:
            all_hidden_states = [
                hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states
            ]
        else:
            all_hidden_states = [
                hidden_states[0].cpu().numpy()
                for hidden_states in all_hidden_states[1:]
            ]
        all_hidden_states = np.array(all_hidden_states, dtype=dtype)

    # print('Sentence         : "%s"' % (sentence))
    # print("Original    (%03d): %s" % (len(original_tokens), original_tokens))
    # print(
    #     "Tokenized   (%03d): %s"
    #     % (
    #         len(tokenizer.convert_ids_to_tokens(ids)),
    #         tokenizer.convert_ids_to_tokens(ids),
    #     )
    # )

    assert all_hidden_states.shape[1] == len(ids)

    if output_level == 'sentence' and sentence_pooling == 'cls':
        final_hidden_states = all_hidden_states[:, 0:1, :]  # (L, 1, H)
        return final_hidden_states, [sentence]
    
    # Handle special tokens
    # filtered_ids will contain all ids if we are extracting with
    #  special tokens, and only normal word/subword ids if we are
    #  extracting without special tokens
    # all_hidden_states will also be filtered at this step to match
    #  the ids in filtered ids
    filtered_ids = ids
    idx_special_tokens = [t_i for t_i, x in enumerate(ids) if x in special_tokens_ids]
    special_token_ids = [ids[t_i] for t_i in idx_special_tokens]

    if not include_special_tokens:
        idx_without_special_tokens = [
            t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids
        ]
        filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
        all_hidden_states = all_hidden_states[:, idx_without_special_tokens, :]
        special_token_ids = []

    assert all_hidden_states.shape[1] == len(filtered_ids)
    # print(
    #     "Filtered   (%03d): %s"
    #     % (
    #         len(tokenizer.convert_ids_to_tokens(filtered_ids)),
    #         tokenizer.convert_ids_to_tokens(filtered_ids),
    #     )
    # )

    # Get actual tokens for filtered ids in order to do subword
    #  aggregation
    segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)

    # Perform subword aggregation/detokenization
    #  After aggregation, we should have |original_tokens| embeddings,
    #  one for each word. If special tokens are included, then we will
    #  have |original_tokens| + |special_tokens|
    counter = 0
    detokenized = []
    final_hidden_states = np.zeros(
        (
            all_hidden_states.shape[0],
            len(original_tokens) + len(special_token_ids),
            all_hidden_states.shape[2],
        ),
        dtype=dtype,
    )
    inputs_truncated = False

    # Keep track of what the previous token was. This is used to detect
    #  special tokens followed/preceeded by dropped tokens, which is an
    #  ambiguous situation for the detokenizer
    prev_token_type = "NONE"

    last_special_token_pointer = 0
    for token_idx, token in enumerate(tmp_tokens):
        # Handle special tokens
        if include_special_tokens and tokenization_counts[token] != 0:
            if last_special_token_pointer < len(idx_special_tokens):
                while (
                    last_special_token_pointer < len(idx_special_tokens)
                    and counter == idx_special_tokens[last_special_token_pointer]
                ):
                    assert prev_token_type != "DROPPED", (
                        "A token dropped by the tokenizer appeared next "
                        + "to a special token. Detokenizer cannot resolve "
                        + f"the ambiguity, please remove '{sentence}' from"
                        + "the dataset, or try a different tokenizer"
                    )
                    prev_token_type = "SPECIAL"
                    final_hidden_states[:, len(detokenized), :] = all_hidden_states[
                        :, counter, :
                    ]
                    detokenized.append(
                        segmented_tokens[idx_special_tokens[last_special_token_pointer]]
                    )
                    last_special_token_pointer += 1
                    counter += 1

        current_word_start_idx = counter
        current_word_end_idx = counter + tokenization_counts[token]

        # Check for truncated hidden states in the case where the
        # original word was actually tokenized
        if (
            tokenization_counts[token] != 0
            and current_word_start_idx >= all_hidden_states.shape[1]
        ) or current_word_end_idx > all_hidden_states.shape[1]:
            final_hidden_states = final_hidden_states[
                :,
                : len(detokenized)
                + len(special_token_ids)
                - last_special_token_pointer,
                :,
            ]
            inputs_truncated = True
            break

        if tokenization_counts[token] == 0:
            assert prev_token_type != "SPECIAL", (
                "A token dropped by the tokenizer appeared next "
                + "to a special token. Detokenizer cannot resolve "
                + f"the ambiguity, please remove '{sentence}' from"
                + "the dataset, or try a different tokenizer"
            )
            prev_token_type = "DROPPED"
        else:
            prev_token_type = "NORMAL"

        final_hidden_states[:, len(detokenized), :] = aggregate_repr(
            all_hidden_states,
            current_word_start_idx,
            current_word_end_idx - 1,
            token_aggregation,
        )
        detokenized.append(
            "".join(segmented_tokens[current_word_start_idx:current_word_end_idx])
        )
        counter += tokenization_counts[token]

    if include_special_tokens:
        while counter < len(segmented_tokens):
            if last_special_token_pointer >= len(idx_special_tokens):
                break

            if counter == idx_special_tokens[last_special_token_pointer]:
                assert prev_token_type != "DROPPED", (
                    "A token dropped by the tokenizer appeared next "
                    + "to a special token. Detokenizer cannot resolve "
                    + f"the ambiguity, please remove '{sentence}' from"
                    + "the dataset, or try a different tokenizer"
                )
                prev_token_type = "SPECIAL"
                final_hidden_states[:, len(detokenized), :] = all_hidden_states[
                    :, counter, :
                ]
                detokenized.append(
                    segmented_tokens[idx_special_tokens[last_special_token_pointer]]
                )
                last_special_token_pointer += 1
            counter += 1

    # print("Detokenized (%03d): %s" % (len(detokenized), detokenized))
    # print("Counter: %d" % (counter))

    if inputs_truncated:
        print("WARNING: Input truncated because of length, skipping check")
    else:
        assert counter == len(filtered_ids)
        assert len(detokenized) == len(original_tokens) + len(special_token_ids)
    # print("===================================================================")

    if output_level == 'sentence' and sentence_pooling == 'mean':
        final_hidden_states = np.mean(final_hidden_states, axis=1, keepdims=True)
        return final_hidden_states, [sentence]

    return final_hidden_states, detokenized


def extract_representations(
    model_desc,
    input_corpus,
    output_file,
    device="cpu",
    token_aggregation="last",
    output_type="json",
    random_weights=False,
    ignore_embeddings=False,
    decompose_layers=False,
    filter_layers=None,
    dtype="float32",
    include_special_tokens=False,
    output_level="token",
    sentence_pooling="mean",
):
    """
    Extract representations for an entire corpus and save them to disk

    Parameters
    ----------
    model_desc : str
        Model description; can either be a model name like ``bert-base-uncased``,
        a comma separated list indicating <model>,<tokenizer> (since 1.0.8),
        or a path to a trained model

    input_corpus : str
        Path to the input corpus, where each sentence is on its separate line

    output_file : str
        Path to output file. Supports all filetypes supported by
        ``data.writer.ActivationsWriter``.

    device : str, optional
        Specifies the device (CPU/GPU) on which the extraction should be
        performed. Defaults to 'cpu'

    token_aggregation : {'first', 'last', 'average'}, optional
        Aggregation method for combining subword activations. Defaults to 'last'

    output_type : str, optional
        Explicit definition of output file type if it cannot be derived from the
        ``output_file`` path

    random_weights : bool, optional
        Whether the weights of the model should be randomized. Useful for analyses
        where one needs an untrained model. Defaults to False.

    ignore_embeddings : bool, optional
        Whether the embedding layer should be excluded in the final output, or
        kept with the regular layers. Defaults to False

    decompose_layers : bool, optional
        Whether each layer should have it's own output file, or all layers be saved
        in a single file. Defaults to False, i.e. single file

    filter_layers : str
        Comma separated list of layer indices to save. The format is the same as
        the one accepted by ``data.writer.ActivationsWriter``.

    dtype : str, optional
        Data type in which the activations will be stored. Supports all numpy
        based tensor types. Common values are 'float32' and 'float16'. Defaults
        to 'float16'

    include_special_tokens : bool, optional
        Whether or not to special tokens in the extracted representations.
        Special tokens are tokens not present in the original sentence, but are
        added by the tokenizer, such as [CLS], [SEP] etc.
    """
    print(f"Loading model: {model_desc}")
    model, tokenizer = get_model_and_tokenizer(
        model_desc, device=device, random_weights=random_weights
    )

    print("Reading input corpus")

    def corpus_generator(input_corpus_path):
        with open(input_corpus_path, "r") as fp:
            for line in fp:
                yield line.strip()
            return

    print("Preparing output file")
    writer = ActivationsWriter.get_writer(
        output_file,
        filetype=output_type,
        decompose_layers=decompose_layers,
        filter_layers=filter_layers,
        dtype=dtype,
    )

    total = sum(1 for _ in corpus_generator(input_corpus))

    print("Extracting representations from model")
    tokenization_counts = {}  # Cache for tokenizer rules
    with tqdm(total=total, desc=f"Extracting representations for {input_corpus.split('/')[-1]} with {model_desc}", dynamic_ncols=True, leave=False) as pbar:
        for sentence_idx, sentence in enumerate(corpus_generator(input_corpus)):
            hidden_states, extracted_words = extract_sentence_representations(
                sentence,
                model,
                tokenizer,
                device=device,
                include_embeddings=(not ignore_embeddings),
                token_aggregation=token_aggregation,
                dtype=dtype,
                include_special_tokens=include_special_tokens,
                tokenization_counts=tokenization_counts,
                output_level=output_level,     
                sentence_pooling=sentence_pooling,    
            )

            writer.write_activations(sentence_idx, extracted_words, hidden_states)
            pbar.update(1)

    writer.close()

