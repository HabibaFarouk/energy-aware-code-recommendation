# Energy Cost vs. Predictive Benefit of Machine Learning Models for Energy-Aware Code Recommendation

Bachelor thesis project at HTW Berlin, Faculty 2: Ingenieurwissenschaften – Technik und Leben, Betriebswirtschaft/Umweltinformatik.

## Project goal

This repository studies whether machine-learning models can look at two functionally equivalent implementations of the same programming task and predict which one uses less energy. The thesis compares the energy cost of training and running the recommender with the energy savings that a correct recommendation can deliver.

The central question is:

What is the trade-off between the energy spent on model training and inference and the energy that the recommendation can save in practice?

The project addresses three main research questions:

- RQ1: How much energy do the models consume during training and inference?
- RQ2: What is the trade-off between predictive quality and energy use?
- RQ3: How many correct recommendations are needed before the energy saved outweighs the energy spent?

## What the project actually implements

This repository is a working research workflow, not a general-purpose app. The current implementation includes:

- loading and cleaning the paired benchmark data from the Energy-Smells dataset
- feature extraction from Python source using `ast` and `tokenize`
- training data construction with pairwise comparisons and randomization to avoid positional bias
- grouped cross-validation by programming problem to prevent leakage
- models for binary recommendation:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - majority-class baseline
  - frozen CodeBERT-based embedding experiment
- feature configurations:
  - Config A: per-version static features
  - Config B: feature differences
  - Combined: per-version + difference features
- energy/resource measurement for training and inference using software-based monitoring and optional CodeCarbon/manual power-meter entries
- comparison of model quality vs. measured energy cost and break-even analysis
- final analysis notebooks and saved result artifacts

## Methodology

The project uses the public Energy-Smells benchmark dataset, which contains paired Python implementations of the same programming task, together with measured energy, runtime, and memory values for both versions. The dataset contains 21,428 pairs across 1,232 programming problems.

The classification task is to predict whether the first or second implementation uses less energy. The target label is:

- `energy_winner = 1`: V1 uses less energy
- `energy_winner = 0`: V0 uses less energy

Equal-energy pairs are treated as ties and excluded from training.

The model inputs are static source-code features such as:

- program length
- number of loops
- nesting depth
- conditionals
- function calls
- imports
- other AST-derived counts and code-structure proxies

Measured energy, runtime, and memory are not used as model features because they would leak the label.

## Repository structure

- `Notebooks/`: analysis and modeling pipeline
- `Data/Raw/`: raw benchmark inputs
- `Data/Processed/`: cleaned and feature-engineered datasets and trained model metadata
- `Results/`: evaluation metrics, plots, and measurement outputs
- `experiment_utils.py`: training and evaluation helpers
- `measurement_utils.py`: energy/resource measurement utilities

## Notebook workflow

Run the notebooks from the repository root in order:

1. [01_data_understanding.ipynb](Notebooks/01_data_understanding.ipynb)  
   Loads the raw paired benchmark files, validates quality, and writes cleaned data.

2. [02_exploratory_analysis.ipynb](Notebooks/02_exploratory_analysis.ipynb)  
   Summarizes energy, runtime, and memory outcomes and problem-level patterns.

3. [03_feature_engineering.ipynb](Notebooks/03_feature_engineering.ipynb)  
   Extracts static features, builds the training dataset, and prepares pairwise winner labels.

4. [04_model_training.ipynb](Notebooks/04_model_training.ipynb)  
   Trains and evaluates Logistic Regression, Random Forest, and XGBoost on both prepared datasets using problem-grouped cross-validation, then measures training and held-out inference resources in the same notebook execution. Training and inference results are saved separately.

5. [05_codebert_energy_measurements.ipynb](Notebooks/05_codebert_energy_measurements.ipynb)  
   Trains frozen CodeBERT embeddings with a Logistic Regression head in separate cells for the validated and significant datasets, measuring each training workload with runtime, CPU, memory, available hardware-energy readings, and CodeCarbon estimates when installed. Separate inference cells measure 100 held-out pairs for 10 repetitions and explicitly extrapolate energy per pair to each dataset's pair count. Inputs are capped at 512 tokens, including special tokens; standard CodeBERT positional embeddings are not extended. Each run also creates an external power-meter entry template. CodeBERT resource results are kept separate from tabular-model measurements.

6. [06_final_analysis.ipynb](Notebooks/06_final_analysis.ipynb)  
   Reads the saved artifacts for both datasets and model families and evaluates predictive quality, measured energy trade-offs, and break-even points. It distinguishes direct sensor measurements from unavailable energy readings and linear inference extrapolations.

`Notebooks/07_codebert_1024_experiment.ipynb` is an archival note only. Its former experiment changed pretrained positional embeddings and is not part of the current pipeline. Existing `codebert_1024` results and model files are historical and must not be interpreted as current 512-token results.

## Data sources and labels

The raw inputs in `Data/Raw/` include:

- `public_validated_energy_results.jsonl`
- `public_significant_energy_diff.jsonl`

The project also recognizes an optional classified-smells dataset, but it is not required for the core recommendation workflow.

The main processed results include:

- cleaned paired benchmark tables
- feature-engineered training datasets
- fold-level model metrics
- out-of-fold predictions
- final analysis CSVs and charts

## Results and artifacts

The project stores generated artifacts in `Results/` and `Data/Processed/`:

- dataset summaries and quality reports
- feature-engineering summaries and manifests
- model performance tables by model and configuration
- fold-level metrics and predictions
- feature importance outputs and ablation experiments
- large energy-difference analysis
- model energy impact vs. baseline comparison
- break-even analysis
- resource/energy measurement summaries and per-run folders

This repository therefore implements the full thesis workflow from data preparation to model comparison and energy trade-off analysis.

## Important note for interpretation

The results are meant to answer a research question, not to provide a production-ready recommender system. The models predict which of two implementations is likely lower-energy; they do not rewrite code or generate an optimized program. The measured resource cost is reported separately from the benchmark outcome analysis and must be interpreted with the selected dataset, feature configuration, and measurement metadata.

## Status

The implemented workflow is already present in the repository and includes the data-cleaning, feature-engineering, model-training, measurement, and final-analysis notebooks and outputs described above.
