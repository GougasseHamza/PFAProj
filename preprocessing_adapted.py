"""
Preprocessing for PhysioNet Eye Tracking ECG Dataset.
Converts AOI-based gaze data into symbolic scanpath sequences.
"""

import pandas as pd
import numpy as np
import os
import sys


STANDARD_LEADS = {'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}

RHYTHM_SEGMENTS = {'1', '2', '3'}

EXPERTISE_MAPPING = {
    'Consultant': 'Expert',
    'Fellow': 'Expert',
    'resident': 'Intermediate',
    'General Doctor': 'Intermediate',
    'Technician': 'Intermediate',
    'Nurse': 'Intermediate',
    'CCU Nurse': 'Intermediate',
    'Cathlab Nurse': 'Intermediate',
    'Med 1': 'Novice',
    'Med 2': 'Novice'
}

FIXATION_THRESHOLD_MS = 200


def extract_lead_from_aoi(aoi_label):
    """Map an AOI label like 'V5-1 NSR' to a lead name like 'V5'."""
    if not isinstance(aoi_label, str) or not aoi_label.strip():
        return None

    if 'Information' in aoi_label:
        return None

    token = aoi_label.split()[0]
    base = token.split('-')[0]

    if base in STANDARD_LEADS:
        return base
    if base in RHYTHM_SEGMENTS:
        return 'Rh'
    return None


def run_length_encode(symbols):
    """Collapse consecutive identical symbols: [A, A, B] -> [A, B]."""
    if not symbols:
        return []
    encoded = [symbols[0]]
    for s in symbols[1:]:
        if s != encoded[-1]:
            encoded.append(s)
    return encoded


def create_scanpaths(aoi_df, threshold_ms=FIXATION_THRESHOLD_MS):
    """Convert filtered AOI data into symbolic scanpath sequences."""
    scanpaths = []

    grouped = aoi_df.groupby(['Respondent_Name', 'ParentStimulus', 'Group'])
    for (participant, stimulus, group), gdf in grouped:
        gdf = gdf.sort_values('Hit_time_G')

        symbols = []
        for _, row in gdf.iterrows():
            lead = extract_lead_from_aoi(row['Label'])
            if lead is None:
                continue
            avg_dur = row['Average_Fixations_Duration']
            if pd.notna(avg_dur):
                suffix = 's' if avg_dur < threshold_ms else 'l'
                symbols.append(f"{lead}_{suffix}")

        if len(symbols) < 3:
            continue

        compressed = run_length_encode(symbols)
        expertise = EXPERTISE_MAPPING.get(group, 'Unknown')

        scanpaths.append({
            'participant': participant,
            'ecg': stimulus,
            'group': group,
            'expertise': expertise,
            'scanpath': compressed,
            'scanpath_string': ' '.join(compressed),
            'length': len(compressed),
            'raw_length': len(symbols)
        })

    return pd.DataFrame(scanpaths)


def process_physionet_dataset(input_file, output_dir='data', expertise_filter=None):
    """Load, filter, and convert the PhysioNet dataset to scanpaths."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PhysioNet ECG Eye Tracking Dataset Preprocessing")
    print("=" * 70)

    print("\nLoading dataset...")
    df = pd.read_csv(input_file)
    aoi_df = df[df['Type'] == 'Static AOI'].copy()

    print(f"  Total rows: {len(df)}")
    print(f"  AOI entries: {len(aoi_df)}")
    print(f"  Participants: {aoi_df['Respondent_Name'].nunique()}")
    print(f"  ECGs: {aoi_df['ParentStimulus'].nunique()}")

    print("\nCreating symbolic scanpaths...")
    scanpaths_df = create_scanpaths(aoi_df)
    print(f"  Generated {len(scanpaths_df)} scanpaths")

    print("\nExpertise distribution:")
    for expertise, count in scanpaths_df['expertise'].value_counts().items():
        print(f"  {expertise}: {count}")

    if expertise_filter:
        scanpaths_df = scanpaths_df[scanpaths_df['expertise'] == expertise_filter].copy()
        print(f"\n  Filtered to {len(scanpaths_df)} {expertise_filter} scanpaths")

    print(f"\nScanpath stats: avg={scanpaths_df['length'].mean():.1f}, "
          f"min={scanpaths_df['length'].min()}, max={scanpaths_df['length'].max()}")

    return scanpaths_df


def create_train_test_split(scanpaths_df, test_size=0.2, output_dir='data'):
    """Split expert and novice scanpaths into train/test sets."""
    os.makedirs(output_dir, exist_ok=True)

    expert_df = scanpaths_df[scanpaths_df['expertise'] == 'Expert'].copy()
    novice_df = scanpaths_df[scanpaths_df['expertise'] == 'Novice'].copy()

    print(f"\nDataset: {len(expert_df)} expert, {len(novice_df)} novice")

    n_exp_test = int(len(expert_df) * test_size)
    expert_df = expert_df.sample(frac=1, random_state=42).reset_index(drop=True)
    expert_train = expert_df.iloc[:-n_exp_test] if n_exp_test > 0 else expert_df
    expert_test = expert_df.iloc[-n_exp_test:] if n_exp_test > 0 else pd.DataFrame()

    n_nov_test = int(len(novice_df) * test_size)
    novice_df = novice_df.sample(frac=1, random_state=42).reset_index(drop=True)
    novice_train = novice_df.iloc[:-n_nov_test] if n_nov_test > 0 else novice_df
    novice_test = novice_df.iloc[-n_nov_test:] if n_nov_test > 0 else pd.DataFrame()

    print(f"Split: expert {len(expert_train)}/{len(expert_test)}, "
          f"novice {len(novice_train)}/{len(novice_test)} (train/test)")

    expert_train.to_csv(f'{output_dir}/expert_train.csv', index=False)
    expert_test.to_csv(f'{output_dir}/expert_test.csv', index=False)
    novice_train.to_csv(f'{output_dir}/novice_train.csv', index=False)
    novice_test.to_csv(f'{output_dir}/novice_test.csv', index=False)
    scanpaths_df.to_csv(f'{output_dir}/scanpaths_all.csv', index=False)

    return expert_train, expert_test, novice_train, novice_test


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "Grid_Anonymized.csv"

    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' not found")
        sys.exit(1)

    all_scanpaths = process_physionet_dataset(input_file, output_dir='data')
    create_train_test_split(all_scanpaths, test_size=0.2, output_dir='data')
    print("\nDone!")
