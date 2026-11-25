# Energy Cost vs. Predictive Benefit of Machine Learning Models for Energy-Aware Code Recommendation

Bachelor thesis project at HTW Berlin, Faculty 2: Ingenieurwissenschaften – Technik und Leben, Betriebliche Umweltinformatik.

## Project goal

This repository studies whether machine-learning models can look at two functionally equivalent implementations of the same programming task and predict which one uses less energy. The thesis compares the energy cost of training and running the recommender with the energy savings that a correct recommendation can deliver.

The central question is:

What is the trade-off between the energy spent on model training and inference and the energy that the recommendation can save in practice?

The project addresses three main research questions:

- RQ1: How much energy do the models consume during training and inference?
- RQ2: What is the trade-off between predictive quality and energy use?
- RQ3: How many correct recommendations are needed before the energy saved outweighs the energy spent?

## Where things run: laptop vs. testbed

This project runs in two places, and it is important to keep them separate:

- **Your laptop** runs every notebook that produces the *predictive-quality* results
  (accuracy, balanced accuracy, ROC-AUC, F1). These numbers are deterministic given a
  fixed random seed, so they never need to be repeated to be trustworthy.
- **The LRZ testbed** (baltikum/kaunas) runs the `scripts/` folder, which repeats
  training and inference several times to get a *statistically reliable energy
  measurement*. Energy and timing have real measurement noise (background
  processes, thermal effects, OS scheduling), so these repetitions average that
  noise out. The testbed also gives real hardware energy readings via Intel RAPL
  counters, instead of the software estimate (CodeCarbon) a laptop falls back to.

Repeating training does **not** change accuracy — it only makes the energy numbers
more reliable. See `scripts/measure_tabular_models.py` and
`scripts/measure_codebert.py` for the testbed-side code, and their docstrings for
why RAPL hardware counters work automatically there.

## What the project actually implements

This repository is a working research workflow, not a general-purpose app. The current implementation includes:

- loading and cleaning the paired benchmark data from the Energy-Smells dataset
- feature extraction from Python source using `ast` and `tokenize`
- training data construction with pairwise comparisons and randomization to avoid positional bias (and a dedicated check that measures accuracy before vs. after this randomization)
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
- energy/resource measurement for training and inference, repeated multiple times for reliability, using real hardware RAPL counters on the testbed (or CodeCarbon's software estimate as a fallback)
- comparison of model quality vs. measured energy cost and break-even analysis
- final analysis notebooks and saved result artifacts

## Methodology

The project uses the public Energy-Smells benchmark dataset, which contains paired Python implementations of the same programming task, together with measured energy, runtime, and memory values for both versions. The dataset contains 21,428 pairs across 1,232 programming problems.

The classification task is to predict whether the first or second implementation uses less energy. The target label is:

- `energy_winner = 1`: V1 uses less energy
- `energy_winner = 0`: V0 uses less energy

Equal-energy pairs are treated as ties and excluded from training. Each pair's V0/V1 orientation is randomly swapped with probability 0.5 before training, so a model cannot win by learning "the version listed second tends to use less energy" instead of learning from the code itself. `Notebooks/04b_swap_sensitivity_check.ipynb` explicitly measures how much accuracy would have looked inflated without this swap.

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

- `Notebooks/`: analysis and modeling pipeline (runs on your laptop)
- `scripts/`: standalone energy-measurement scripts (run on the LRZ testbed)
- `Data/Raw/`: raw benchmark inputs
- `Data/Processed/`: cleaned and feature-engineered datasets and trained model metadata
- `Results/`: evaluation metrics, plots, and measurement outputs
- `experiment_utils.py`: training and evaluation helpers (shared by notebooks and testbed scripts)
- `measurement_utils.py`: energy/resource measurement utilities (shared by notebooks and testbed scripts)

## Notebook workflow (run on your laptop, in order)

1. [01_data_understanding.ipynb](Notebooks/01_data_understanding.ipynb)
   Loads the raw paired benchmark files, validates quality, and writes cleaned data.

2. [02_exploratory_analysis.ipynb](Notebooks/02_exploratory_analysis.ipynb)
   Summarizes energy, runtime, and memory outcomes and problem-level patterns.

3. [03_feature_engineering.ipynb](Notebooks/03_feature_engineering.ipynb)
   Extracts static features, randomly swaps each pair's V0/V1 orientation, and builds the training dataset with pairwise winner labels.

4. [04_model_training_measurment.ipynb](Notebooks/04_model_training_measurment.ipynb)
   **Cells 1–8: quality only.** Trains and evaluates Logistic Regression, Random Forest, and XGBoost on both prepared datasets using problem-grouped cross-validation. Run these cells on your laptop to get accuracy/ROC-AUC — no repetition needed, this is deterministic.
   **Cells 9–10: local energy measurement (optional/legacy).** Measures training and inference resources using software-based monitoring (CodeCarbon). This still works, but the testbed scripts (see below) give more trustworthy hardware-measured numbers and should be treated as the primary source for the thesis's reported energy figures.

5. [04b_swap_sensitivity_check.ipynb](Notebooks/04b_swap_sensitivity_check.ipynb)
   Measures accuracy **before** (original, un-swapped V0/V1 order) and **after** (the randomized order used everywhere else) swapping, for the same models and cross-validation. Added per supervisor feedback.

6. [05b_codebert_quality_only.ipynb](Notebooks/05b_codebert_quality_only.ipynb)
   Trains the frozen CodeBERT embedding model with a Logistic Regression head **once** per dataset, to get accuracy/ROC-AUC quickly without the cost of repeated embedding extraction. Use this for CodeBERT's quality numbers on your laptop.

7. [05_codebert_training_measurements.ipynb](Notebooks/05_codebert_training_measurements.ipynb)
   The full repeated-measurement version of step 6: trains CodeBERT embeddings 3 times (validated) / 10 times (significant), measuring runtime, CPU, memory, and available hardware-energy readings each time. This is slow (full embedding extraction repeated 3x/10x) and, like step 4's Cells 9-10, is superseded by the testbed script `scripts/measure_codebert.py` for the thesis's final reported numbers; keep this notebook mainly as a reference/fallback if testbed access is delayed.

8. [06_final_analysis.ipynb](Notebooks/06_final_analysis.ipynb)
   Reads the saved artifacts for both datasets and model families and evaluates predictive quality, measured energy trade-offs, and break-even points. It distinguishes direct sensor measurements from unavailable energy readings and linear inference extrapolations.

`Notebooks/07_codebert_1024_experiment.ipynb` is an archival note only. Its former experiment changed pretrained positional embeddings and is not part of the current pipeline. Existing `codebert_1024` results and model files are historical and must not be interpreted as current 512-token results.

## Testbed scripts (run on LRZ baltikum/kaunas, not your laptop)

These reuse the exact same `measurement_utils.py` / `experiment_utils.py` functions as the notebooks above — only where they run differs.

- [scripts/measure_tabular_models.py](scripts/measure_tabular_models.py)
  Measures Logistic Regression, Random Forest, and XGBoost training/inference energy, repeated 10 times per model/feature-configuration/dataset, per supervisor feedback.

- [scripts/measure_codebert.py](scripts/measure_codebert.py)
  Measures CodeBERT training/inference energy: 3 repetitions on the validated dataset (the larger, whole dataset), 10 repetitions on the significant dataset (the smaller one), per supervisor feedback.

- [scripts/run_testbed_measurements.sh](scripts/run_testbed_measurements.sh)
  The bash file to pass to `pos commands launch nodeA --infile scripts/run_testbed_measurements.sh`. Sets up a Python virtual environment, checks whether RAPL hardware energy counters are visible on the node, and runs both measurement scripts in sequence.

- [scripts/requirements.txt](scripts/requirements.txt)
  Starter dependency list for the testbed's Python environment. Replace with the exact versions from your laptop (`pip freeze`) before a real measurement run, so the testbed measures the same library versions used for the quality results.

**Prerequisite:** the testbed scripts expect `Data/Processed/` to already exist — i.e. Notebooks 01–03 must be run locally first, and the resulting project folder (including `Data/Processed/`) copied onto the testbed node before launching these scripts.

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
- swap-sensitivity (before/after) comparison
- resource/energy measurement summaries and per-run folders (from both local notebook runs and testbed script runs — keep these distinguishable, since local numbers are software estimates and testbed numbers are hardware-measured)

This repository therefore implements the full thesis workflow from data preparation to model comparison and energy trade-off analysis.

## Important note for interpretation

The results are meant to answer a research question, not to provide a production-ready recommender system. The models predict which of two implementations is likely lower-energy; they do not rewrite code or generate an optimized program. The measured resource cost is reported separately from the benchmark outcome analysis and must be interpreted with the selected dataset, feature configuration, and measurement metadata, and with whether it came from a laptop's software estimate or the testbed's hardware measurement.

## Status

The implemented workflow is already present in the repository and includes the data-cleaning, feature-engineering, swap-sensitivity, model-training, measurement, testbed-script, and final-analysis notebooks/scripts and outputs described above.
