"""
Main pipeline for ECG Scanpath PFA Analysis.
Runs preprocessing, PFA training, and evaluation on the PhysioNet dataset.

Usage:
    python main_adapted.py --all            Run complete pipeline
    python main_adapted.py --preprocess     Preprocess data only
    python main_adapted.py --train          Train PFA models only
    python main_adapted.py --evaluate       Evaluate models only
"""

import argparse
import os
import numpy as np
import preprocessing_adapted as preprocessing
import pfa_model


def setup_directories():
    for d in ['data', 'models', 'results']:
        os.makedirs(d, exist_ok=True)


def preprocess_data(input_file='Grid_Anonymized.csv'):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return False

    all_scanpaths = preprocessing.process_physionet_dataset(input_file, output_dir='data')
    preprocessing.create_train_test_split(all_scanpaths, test_size=0.2, output_dir='data')
    return True


def train_models():
    expert_file = "data/expert_train.csv"
    novice_file = "data/novice_train.csv"

    if not os.path.exists(expert_file) or not os.path.exists(novice_file):
        print("Training data not found. Run --preprocess first.")
        return None, None

    print("\nTraining Expert PFA...")
    exp_model = pfa_model.train_pfa_model(expert_file, "models/expert_pfa.pkl", visualize=True)

    print("\nTraining Novice PFA...")
    nov_model = pfa_model.train_pfa_model(novice_file, "models/novice_pfa.pkl")

    return exp_model, nov_model


def evaluate_models(exp_model=None, nov_model=None):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    expert_test_file = "data/expert_test.csv"
    novice_test_file = "data/novice_test.csv"

    if not os.path.exists(expert_test_file) or not os.path.exists(novice_test_file):
        print("Test files not found. Run --preprocess first.")
        return

    expert_test = pfa_model.load_scanpaths_from_csv(expert_test_file)
    novice_test = pfa_model.load_scanpaths_from_csv(novice_test_file)

    print(f"\nTest set: {len(expert_test)} expert, {len(novice_test)} novice")

    all_seqs = expert_test + novice_test
    y_true = np.array([1] * len(expert_test) + [0] * len(novice_test))

    scores = []
    for seq in all_seqs:
        n = len(seq)
        log_exp = exp_model.compute_log_likelihood(seq) / n
        log_nov = nov_model.compute_log_likelihood(seq) / n
        scores.append(log_exp - log_nov)
    scores = np.array(scores)

    best_f1, best_tau, best_metrics = -1.0, None, None
    for tau in np.unique(scores):
        y_pred = (scores >= tau).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            try:
                auc = roc_auc_score(y_true, scores)
            except ValueError:
                auc = 0.0
            best_metrics = (acc, prec, rec, f1, auc)

    if best_metrics is None:
        print("Could not compute metrics.")
        return

    acc, prec, rec, f1, auc = best_metrics
    print(f"\nResults (threshold={best_tau:.4f}):")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1-score : {f1:.3f}")
    print(f"  AUC-ROC  : {auc:.3f}")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc_roc': auc}


def run_complete_pipeline():
    print("=" * 70)
    print("ECG SCANPATH PFA ANALYSIS PIPELINE")
    print("=" * 70)

    setup_directories()

    if not os.path.exists("Grid_Anonymized.csv"):
        print("Error: Grid_Anonymized.csv not found in current directory")
        return

    if not preprocess_data():
        return

    exp_model, nov_model = train_models()
    if exp_model is None or nov_model is None:
        return

    evaluate_models(exp_model, nov_model)
    print("\nPipeline complete!")


def main():
    parser = argparse.ArgumentParser(description='ECG Scanpath PFA Analysis')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess dataset')
    parser.add_argument('--train', action='store_true', help='Train PFA models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--input', type=str, default='Grid_Anonymized.csv')
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    setup_directories()

    if args.all:
        run_complete_pipeline()
    else:
        if args.preprocess:
            preprocess_data(args.input)
        if args.train:
            train_models()
        if args.evaluate:
            evaluate_models()


if __name__ == "__main__":
    main()
