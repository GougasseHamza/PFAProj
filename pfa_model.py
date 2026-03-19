"""
Probabilistic Finite Automaton (PFA) for ECG scanpath modeling.
Uses AALpy's Alergia algorithm to learn Markov Chain automata.
"""

import math
import pickle
import pandas as pd
import numpy as np

try:
    from aalpy.learning_algs import run_Alergia
    from aalpy.utils import visualize_automaton
    AALPY_AVAILABLE = True
except ImportError:
    print("WARNING: AALpy not installed. Install with: pip install aalpy")
    AALPY_AVAILABLE = False


class PFAModel:
    """PFA learned via Alergia for modeling ECG scanpath sequences."""

    def __init__(self, eps=0.05):
        self.eps = eps
        self.automaton = None
        self.alphabet = set()
        self.trained = False

    def train(self, scanpaths, verbose=True):
        """Train on a list of scanpath sequences using Alergia."""
        if not AALPY_AVAILABLE:
            raise ImportError("AALpy is required. Install with: pip install aalpy")

        self.alphabet = set()
        for seq in scanpaths:
            self.alphabet.update(seq)

        if verbose:
            print(f"Training PFA: {len(scanpaths)} sequences, "
                  f"{len(self.alphabet)} symbols, eps={self.eps}")

        data = [['START'] + seq for seq in scanpaths]

        self.automaton = run_Alergia(
            data=data,
            automaton_type='mc',
            eps=self.eps,
            print_info=verbose
        )
        self.trained = True

        if verbose:
            print(f"Learned {len(self.automaton.states)} states")

    def compute_log_likelihood(self, scanpath, smoothing=1e-10):
        """Compute log P(w|M) with smoothing for unseen transitions."""
        if not self.trained or not scanpath:
            return float('-inf')

        log_prob = 0.0
        state = self.automaton.initial_state

        for symbol in scanpath:
            found = False
            for next_state, prob in state.transitions:
                if next_state.output == symbol:
                    log_prob += math.log(max(prob, smoothing))
                    state = next_state
                    found = True
                    break
            if not found:
                log_prob += math.log(smoothing)

        return log_prob

    def compute_likelihood(self, scanpath):
        """Compute P(w|M)."""
        ll = self.compute_log_likelihood(scanpath)
        return math.exp(ll) if ll != float('-inf') else 0.0

    def compute_perplexity(self, scanpath):
        """Compute perplexity PP(w) = P(w)^(-1/|w|)."""
        p = self.compute_likelihood(scanpath)
        if p <= 0 or len(scanpath) == 0:
            return float('inf')
        return p ** (-1.0 / len(scanpath))

    def classify(self, scanpath, threshold=4.2):
        """Classify as Expert or Novice based on perplexity threshold."""
        pp = self.compute_perplexity(scanpath)
        label = "Expert" if pp < threshold else "Novice"
        return label, pp

    def visualize(self, output_path="pfa_model.pdf"):
        """Save automaton visualization to PDF."""
        if not self.trained or not AALPY_AVAILABLE:
            return
        try:
            visualize_automaton(self.automaton, path=output_path)
            print(f"Saved visualization to {output_path}")
        except Exception as e:
            print(f"Visualization failed: {e}")

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'eps': self.eps, 'alphabet': list(self.alphabet),
                         'trained': self.trained}, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.eps = data['eps']
        self.alphabet = set(data['alphabet'])
        self.trained = data['trained']


def load_scanpaths_from_csv(filepath):
    """Load scanpath strings from CSV and return as list of symbol lists."""
    df = pd.read_csv(filepath)
    scanpaths = []
    for s in df['scanpath_string']:
        if pd.notna(s) and s.strip():
            symbols = s.split()
            if symbols:
                scanpaths.append(symbols)
    return scanpaths


def train_pfa_model(scanpaths_file, output_model=None, visualize=False):
    """Train a PFA model from a scanpaths CSV file."""
    scanpaths = load_scanpaths_from_csv(scanpaths_file)
    print(f"Loaded {len(scanpaths)} scanpaths from {scanpaths_file}")

    if not scanpaths:
        raise ValueError("No valid scanpaths found")

    model = PFAModel(eps=0.05)
    model.train(scanpaths, verbose=True)

    if output_model:
        model.save(output_model)
        print(f"Model saved to {output_model}")

    if visualize and AALPY_AVAILABLE:
        model.visualize("pfa_automaton.pdf")

    return model


if __name__ == "__main__":
    import os
    if os.path.exists("data/expert_train.csv"):
        train_pfa_model("data/expert_train.csv", "models/expert_pfa.pkl", visualize=True)
    else:
        print("Run preprocessing first.")
