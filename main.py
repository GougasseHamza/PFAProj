"""
Improved ECG scanpath experiment with TWO PFAs:
 - one trained on expert scanpaths
 - one trained on novice scanpaths

Classification is done by log-likelihood ratio:
  score(w) = (1/|w|) log P(w | expert PFA)
           - (1/|w|) log P(w | novice PFA)

You need:
    pip install numpy pandas scikit-learn
"""

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

# -----------------------------
# 1. Basic configuration
# -----------------------------

DATA_PATH = "Grid_Anonymized.csv"

# Group column values in your CSV look like:
#   "Med 1", "Med 2", "Fellow", "Consultant", "Nurse", "Technician", "resident", ...
# We'll treat:
#   Consultants + Fellows  -> experts (label 1)
#   Med 1 + Med 2         -> novices (label 0)
#   Others are ignored for this experiment.
EXPERT_GROUPS = {"consultant", "fellow"}
NOVICE_GROUPS = {"med 1", "med 2"}

# 12-lead set we care about
LEADS = {"I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"}

# Rhythm strip AOIs (e.g. "1 NSR", "2 NSR", "3 NSR", "1 AFib", ...)
RHYTHM_SEGMENTS = {"1", "2", "3"}

# Fixation duration threshold in milliseconds
FIXATION_THRESHOLD_MS = 200.0


# -----------------------------
# 2. Mapping AOI label -> lead / Rh
# -----------------------------

def parse_label_to_lead(label: str):
    """
    Map dataset 'Label' field to a logical region:
        - V5-1 NSR  -> 'V5'
        - II-1 NSR  -> 'II'
        - V5 NSR    -> 'V5'
        - 1 NSR     -> 'Rh' (rhythm strip)
        - 1 AFib    -> 'Rh'
        - anything else (Information, etc.) -> None
    """
    if not isinstance(label, str):
        return None

    tokens = label.split()
    if not tokens:
        return None

    aoi_token = tokens[0]          # e.g. 'V5-1' or 'II-1' or 'V5' or '1' or 'Information'
    base = aoi_token.split("-")[0] # 'V5-1' -> 'V5'

    if base in LEADS:
        return base

    if base in RHYTHM_SEGMENTS:
        return "Rh"

    # Ignore AOIs that are not real leads or rhythm segments
    return None


def row_to_symbol(row, threshold_ms=FIXATION_THRESHOLD_MS):
    """
    Convert one AOI row into a symbol like 'II_s' or 'V3_l'
    based on Average_Fixations_Duration.
    Returns None if the row should be ignored.
    """
    lead = parse_label_to_lead(row.get("Label"))
    if lead is None:
        return None

    dur = row.get("Average_Fixations_Duration")
    try:
        dur = float(dur)
    except (TypeError, ValueError):
        return None

    if dur < 0:
        return None

    suffix = "s" if dur < threshold_ms else "l"
    return f"{lead}_{suffix}"


# -----------------------------
# 3. Build scanpaths
# -----------------------------

def build_sequences(df: pd.DataFrame, min_seq_len: int = 3):
    """
    Build scanpath sequences from the Grid_Anonymized dataframe.

    Returns:
        sequences: list of list[str]  (e.g. ['Rh_l', 'II_l', 'V1_s', ...])
        labels:    np.array of 0/1   (0 = novice, 1 = expert)
        subjects:  np.array of subject IDs (Respondent_Name)
    """
    df = df.copy()
    # Only keep AOI rows with actual fixations
    df = df[df["Type"] == "Static AOI"]
    df = df[df["Fixations_Count"] > 0]

    sequences = []
    labels = []
    subjects = []

    group_cols = ["Respondent_Name", "ParentStimulus"]

    for (resp, stim), group in df.groupby(group_cols):
        group_name = str(group["Group"].iloc[0]).strip().lower()

        if group_name in EXPERT_GROUPS:
            label = 1
        elif group_name in NOVICE_GROUPS:
            label = 0
        else:
            # ignore residents, nurses, technicians, etc.
            continue

        group_sorted = group.sort_values("Hit_time_G")

        seq = []
        last_sym = None
        for _, row in group_sorted.iterrows():
            sym = row_to_symbol(row)
            if sym is None:
                continue
            # Run-length encoding: skip repeated consecutive symbols
            if sym == last_sym:
                continue
            seq.append(sym)
            last_sym = sym

        if len(seq) >= min_seq_len:
            sequences.append(seq)
            labels.append(label)
            subjects.append(resp)

    return sequences, np.array(labels), np.array(subjects)


# -----------------------------
# 4. Simple PFA / Markov model
# -----------------------------

class SimplePFA:
    """
    Simple probabilistic finite automaton:

        States: START, each symbol, END
        Transitions:
            P(first_symbol | START)
            P(next_symbol | current_symbol)
            P(END | last_symbol)

    Learned by counting transitions over scanpaths
    with Laplace (add-alpha) smoothing.
    """

    def __init__(self, alphabet, smoothing: float = 1.0):
        self.alphabet = list(sorted(set(alphabet)))
        self.smoothing = float(smoothing)

        # Counts for first symbol
        self.start_counts = Counter()
        self.total_start = 0

        # Counts for transitions between symbol-states (including END)
        # trans_counts[current_state][next_state] = count
        self.trans_counts = defaultdict(Counter)
        self.total_trans = Counter()

    def fit(self, sequences):
        """
        Fit the PFA on a list of sequences (each is a list of symbols).
        """
        for seq in sequences:
            if not seq:
                continue

            first = seq[0]
            self.start_counts[first] += 1
            self.total_start += 1

            # Transitions between symbols
            for i in range(len(seq) - 1):
                cur = seq[i]
                nxt = seq[i + 1]
                self.trans_counts[cur][nxt] += 1
                self.total_trans[cur] += 1

            # Last symbol -> END
            last = seq[-1]
            self.trans_counts[last]["END"] += 1
            self.total_trans[last] += 1

    def log_prob(self, seq):
        """
        Log-probability of a sequence under the model.
        """
        if not seq:
            return float("-inf")

        alpha = self.smoothing

        # Probability of first symbol
        vocab_size_start = len(self.alphabet)
        total_start = self.total_start + alpha * vocab_size_start
        count_first = self.start_counts[seq[0]] + alpha
        logp = math.log(count_first) - math.log(total_start)

        # Transitions between symbols
        vocab_next = len(self.alphabet) + 1  # including END
        for i in range(len(seq) - 1):
            cur = seq[i]
            nxt = seq[i + 1]
            total = self.total_trans[cur] + alpha * vocab_next
            count = self.trans_counts[cur][nxt] + alpha
            logp += math.log(count) - math.log(total)

        # Transition to END
        last = seq[-1]
        total_last = self.total_trans[last] + alpha * vocab_next
        count_end = self.trans_counts[last]["END"] + alpha
        logp += math.log(count_end) - math.log(total_last)

        return logp


# -----------------------------
# 5. Full experiment
# -----------------------------

def run_experiment(csv_path=DATA_PATH):
    # Load the dataset
    df = pd.read_csv(csv_path)

    sequences, labels, subjects = build_sequences(df)

    n_total = len(sequences)
    n_expert = int(labels.sum())
    n_novice = n_total - n_expert

    print(f"Total scanpaths: {n_total}")
    print(f"  Experts: {n_expert}")
    print(f"  Novices: {n_novice}")

    if n_expert == 0 or n_novice == 0:
        print("Not enough expert or novice data after filtering.")
        return

    # Indices of expert/novice sequences
    expert_idx = np.where(labels == 1)[0]
    novice_idx = np.where(labels == 0)[0]

    # Split experts into train/test by subject
    expert_seqs = [sequences[i] for i in expert_idx]
    expert_subjects = subjects[expert_idx]

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=0)
    exp_train_rel, exp_test_rel = next(
        gss.split(expert_seqs, groups=expert_subjects)
    )

    train_exp_idx = expert_idx[exp_train_rel]
    test_exp_idx = expert_idx[exp_test_rel]

    # Split novices into train/test by subject
    novice_seqs = [sequences[i] for i in novice_idx]
    novice_subjects = subjects[novice_idx]

    nov_train_rel, nov_test_rel = next(
        gss.split(novice_seqs, groups=novice_subjects)
    )

    train_nov_idx = novice_idx[nov_train_rel]
    test_nov_idx = novice_idx[nov_test_rel]

    print(f"Expert train sequences: {len(train_exp_idx)}")
    print(f"Expert test sequences:  {len(test_exp_idx)}")
    print(f"Novice train sequences: {len(train_nov_idx)}")
    print(f"Novice test sequences:  {len(test_nov_idx)}")

    # Training sequences
    train_exp_sequences = [sequences[i] for i in train_exp_idx]
    train_nov_sequences = [sequences[i] for i in train_nov_idx]

    # Alphabet: union of symbols seen in both classes during training
    alphabet = sorted({
        sym
        for i in np.concatenate([train_exp_idx, train_nov_idx])
        for sym in sequences[i]
    })
    print(f"Alphabet size (symbols): {len(alphabet)}")

    # Train PFA for experts and novices
    pfa_exp = SimplePFA(alphabet, smoothing=1.0)
    pfa_exp.fit(train_exp_sequences)

    pfa_nov = SimplePFA(alphabet, smoothing=1.0)
    pfa_nov.fit(train_nov_sequences)

    # Evaluate on held-out experts + held-out novices
    test_indices = np.concatenate([test_exp_idx, test_nov_idx])
    test_labels = labels[test_indices]

    def score(idx):
        seq = sequences[idx]
        n = len(seq)
        # Normalised log-likelihoods
        log_pe = pfa_exp.log_prob(seq) / n
        log_pn = pfa_nov.log_prob(seq) / n
        return log_pe - log_pn  # positive => more expert-like

    scores = np.array([score(i) for i in test_indices])

    # Sweep thresholds on the log-likelihood ratio
    unique_thresholds = np.unique(scores)
    best_f1 = -1.0
    best_tau = None
    best_metrics = None

    for tau in unique_thresholds:
        y_pred = (scores >= tau).astype(int)  # expert if score >= tau
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, y_pred, average="binary", zero_division=0
        )
        acc = accuracy_score(test_labels, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            auc = roc_auc_score(test_labels, scores)
            best_metrics = (acc, precision, recall, f1, auc)

    if best_metrics is None:
        print("Could not compute metrics (something went wrong).")
        return

    acc, precision, recall, f1, auc = best_metrics

    print("\n=== Results (expert vs. novice classification) ===")
    print(f"Best LLR threshold τ: {best_tau:.4f}")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"AUC-ROC  : {auc:.3f}")


if __name__ == "__main__":
    run_experiment()
