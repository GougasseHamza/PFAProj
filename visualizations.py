"""Plotting utilities for ECG scanpath analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_scanpath_lengths(df, output_path='scanpath_lengths.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'expertise' in df.columns:
        sns.boxplot(data=df, x='expertise', y='length', ax=ax)
        ax.set_title('Scanpath Length by Expertise')
    else:
        sns.histplot(data=df, x='length', bins=20, ax=ax)
        ax.set_title('Scanpath Length Distribution')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_symbol_frequency(df, top_n=20, output_path='symbol_frequency.png'):
    all_symbols = []
    for s in df['scanpath_string']:
        if pd.notna(s):
            all_symbols.extend(s.split())

    top = dict(Counter(all_symbols).most_common(top_n))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(list(top.keys()), list(top.values()), color='steelblue')
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top {top_n} Symbols')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_perplexity_distribution(expert_pp, novice_pp, threshold=4.2,
                                 output_path='perplexity_distribution.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(expert_pp, bins=30, alpha=0.6, label='Expert', color='green', density=True)
    ax.hist(novice_pp, bins=30, alpha=0.6, label='Novice', color='red', density=True)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax.set_xlabel('Perplexity')
    ax.set_ylabel('Density')
    ax.set_title('Perplexity: Expert vs Novice')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path='confusion_matrix.png'):
    from sklearn.metrics import confusion_matrix as cm_func
    cm = cm_func(y_true, y_pred, labels=['Expert', 'Novice'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Expert', 'Novice'],
                yticklabels=['Expert', 'Novice'], ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
