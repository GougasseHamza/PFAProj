# ECG Scanpath Pattern Recognition using Probabilistic Finite Automata

Classifying expert vs novice ECG interpretation strategies from eye-tracking data using PFA and feature-engineered ML.

## Dataset

Uses the [PhysioNet Eye Tracking Dataset](https://physionet.org/content/eye-tracking-ecg/1.0.0/) (N=63 participants, 10 ECGs each, Tobii Pro X2-60 at 60 Hz).

Place `Grid_Anonymized.csv` in the project root directory.

## Installation

```bash
pip install -r requirements.txt
```

Optional (for PFA visualization): install [Graphviz](https://graphviz.org/download/) system-wide.

## Usage

### Baseline PFA Pipeline

```bash
python main_adapted.py --all
```

Runs preprocessing, trains expert/novice PFAs via Alergia, evaluates with log-likelihood ratio.

### Improved Classifier (94.3% accuracy)

```bash
python improved_classifier.py
```

Extracts 184 features (fixation stats, temporal allocation, transition matrices), aggregates per participant, classifies with Random Forest. Evaluated via participant-level Leave-One-Out CV.

### Step by Step

```bash
python main_adapted.py --preprocess    # Convert AOI data to symbolic scanpaths
python main_adapted.py --train         # Train expert and novice PFAs
python main_adapted.py --evaluate      # Evaluate with log-likelihood ratio
```

## Results

| Method | Accuracy | Precision | Recall | F1 | AUC-ROC |
|--------|----------|-----------|--------|-----|---------|
| Baseline PFA (LLR) | 54.3% | 0.543 | 1.000 | 0.704 | 0.478 |
| Improved (RF + features) | **94.3%** | **0.947** | **0.947** | **0.947** | **0.951** |

Top discriminating features: fixation variability (CV), temporal dispersion, lead-specific attention (V6, V2), TTFF consistency, Gini coefficient of attention.

## Project Structure

```
Grid_Anonymized.csv          # PhysioNet dataset (download manually)
main_adapted.py              # Baseline PFA pipeline
preprocessing_adapted.py     # AOI to symbolic scanpath conversion
pfa_model.py                 # PFA model (Alergia via AALpy)
improved_classifier.py       # Feature-engineered classifier (94.3%)
visualizations.py            # Plotting utilities
requirements.txt             # Dependencies
```

## References

- Sqalli Houssaini et al. (2022). PhysioNet Eye Tracking Dataset. Scientific Data 9:547
- Carrasco & Oncina (1994). Alergia algorithm. ICGI 1994
- Muskardin et al. (2021). AALpy library. ATVA 2021
