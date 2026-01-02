# CKA Multi-Scenario Analysis

## Overview
The updated `run_cka_dialects.py` script now supports three different CKA comparison scenarios between MSA and dialectal models:

### Scenarios

1. **Scenario 1: model_msa(da_sentence) vs model_da(da_sentence)**
   - Both models process the same dialectal corpus
   - Original implementation
   - Measures how similarly models represent dialectal text

2. **Scenario 2: model_msa(msa_sentence) vs model_da(msa_sentence)**
   - Both models process the same MSA corpus
   - Measures how similarly models represent standard Arabic text
   - Uses the MSA corpus file specified by `--msa_corpus` argument

3. **Scenario 3: model_msa(msa_sentence) vs model_da(da_sentence)**
   - Cross-corpus comparison
   - MSA model processes MSA sentences, dialectal model processes dialectal sentences
   - Measures representation alignment when models process their "native" text varieties
   - Uses the new `compute_cka_cross_corpus()` method

## Changes Made

### 1. `run_cka_dialects.py`
- Added `--msa_corpus` argument (default: `data/CKA/MADAR_Corpus/MADAR_MSA.txt`)
- Modified main loop to run all three scenarios for each city-model pair
- Enhanced error handling with per-scenario try-except blocks
- Updated output JSON structure to include:
  - `scenario`: Description of the comparison type
  - `corpus_model1`: Corpus used for model 1
  - `corpus_model2`: Corpus used for model 2
- Changed output filename to `cka_results_all_scenarios.json`
- Added graceful degradation: if MSA corpus doesn't exist, only Scenario 1 runs

### 2. `src/cka.py`
- Added new method: `compute_cka_cross_corpus(corpus_path_1, corpus_path_2, ...)`
- Handles different corpora for each model
- Automatically truncates to minimum length if corpora have different sizes
- Maintains same API as `compute_cka()` for consistency

## Usage

### Basic usage (all scenarios):
```bash
python run_cka_dialects.py \
    --msa_model CAMeL-Lab/bert-base-arabic-camelbert-msa \
    --json models_dialect.json \
    --corpus_dir data/CKA/MADAR_Corpus \
    --msa_corpus data/CKA/MADAR_Corpus/MADAR_MSA.txt \
    --out_dir results_cka \
    --device cuda
```

### Only scenario 1 (if MSA corpus unavailable):
```bash
python run_cka_dialects.py \
    --msa_corpus /nonexistent/path.txt
```

### With unbiased HSIC estimator:
```bash
python run_cka_dialects.py --unbiased
```

## Output Format

```json
{
    "msa_model": "CAMeL-Lab/bert-base-arabic-camelbert-msa",
    "msa_corpus": "data/CKA/MADAR_Corpus/MADAR_MSA.txt",
    "unbiased": false,
    "results": [
        {
            "city": "Fes",
            "dialect_model": "SI2M-Lab/DarijaBERT",
            "scenario": "msa_model(da) vs dialect_model(da)",
            "corpus_model1": "data/CKA/MADAR_Corpus/MADAR_Fes.txt",
            "corpus_model2": "data/CKA/MADAR_Corpus/MADAR_Fes.txt",
            "cka": [[...], [...], ...]
        },
        {
            "city": "Fes",
            "dialect_model": "SI2M-Lab/DarijaBERT",
            "scenario": "msa_model(msa) vs dialect_model(msa)",
            "corpus_model1": "data/CKA/MADAR_Corpus/MADAR_MSA.txt",
            "corpus_model2": "data/CKA/MADAR_Corpus/MADAR_MSA.txt",
            "cka": [[...], [...], ...]
        },
        {
            "city": "Fes",
            "dialect_model": "SI2M-Lab/DarijaBERT",
            "scenario": "msa_model(msa) vs dialect_model(da)",
            "corpus_model1": "data/CKA/MADAR_Corpus/MADAR_MSA.txt",
            "corpus_model2": "data/CKA/MADAR_Corpus/MADAR_Fes.txt",
            "cka": [[...], [...], ...]
        }
    ]
}
```

## Research Applications

These three scenarios enable different research questions:

1. **Scenario 1**: How do models differ when processing dialectal text?
2. **Scenario 2**: How do models differ when processing standard Arabic?
3. **Scenario 3**: How aligned are model representations when each processes its intended text variety?

## Notes

- All scenarios use the same `sentence_pooling="mean"` strategy
- CKA matrices are layer×layer comparisons (typically 13×13 for BERT models)
- The script automatically handles missing corpora gracefully
- Cross-corpus comparison (Scenario 3) truncates to the shorter corpus length
