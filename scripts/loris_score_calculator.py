#!/usr/bin/env python
"""
LORIS and LR5 (no TMB) Score Calculator

Generates LORIS scores using PUBLISHED coefficients from Chang et al., Nature Cancer 2024.
For LR5 (no TMB), trains on Chowell data since those coefficients are not published.

Sources:
- LORIS Paper: https://www.nature.com/articles/s43018-024-00772-7
- PMC Full Text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11962634/
- LORIS Web Tool: https://loris.ccr.cancer.gov
- LORIS GitHub: https://github.com/rootchang/LORIS
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = Path('examples/data')
# The directory is created lazily in main() right before the files are written,
# so importing this module (e.g. from the vignette or the smoke test) has no
# filesystem side effect and never leaves an empty output/loris/ behind.
OUTPUT_DIR = Path('output/loris')

# Preprocessing constants (from LORIS paper)
TMB_CAP = 50
NLR_CAP = 25
AGE_CAP = 85
ALBUMIN_IMPUTE = 3.97  # Median from Chowell train set (per LORIS paper)
NLR_IMPUTE = 6.2       # Median from Chowell train set

# =============================================================================
# PUBLISHED LORIS Coefficients (LLR6 model)
# From PMC Full Text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11962634/
# Table in Methods section: "Six-variable model coefficients"
# =============================================================================
LORIS_COEFFICIENTS = {
    'intercept': -2.0886,
    'TMB': 0.0371,       # TMB (capped at 50)
    'PSTH': -0.8775,     # Prior systemic therapy history (0 or 1)
    'Albumin': 0.5382,   # Albumin (g/dL)
    'NLR': -0.0330,      # Neutrophil-to-lymphocyte ratio (capped at 25)
    'Age': 0.0049,       # Age (capped at 85)
}

# Cancer type calibration terms (from LORIS paper Supplementary Table)
# These adjust the baseline for different cancer types
CANCER_TYPE_CALIBRATION = {
    'Bladder': -0.3323,
    'Breast': -0.2632,
    'CUP': -0.4388,           # Cancer of Unknown Primary
    'Colorectal': -0.6802,
    'Esophagogastric': -0.7079,
    'Glioma': -0.2981,
    'HNSCC': -0.2015,         # Head and Neck Squamous Cell Carcinoma
    'HCC': -0.1863,           # Hepatocellular Carcinoma
    'Melanoma': -0.0086,
    'Mesothelioma': 0.4017,
    'NSCLC': 0.0008,          # Non-Small Cell Lung Cancer
    'SCLC': -0.1219,          # Small Cell Lung Cancer
    'Ovarian': 0.2411,
    'Pancreatic': -1.1813,
    'RCC': 0.5451,            # Renal Cell Carcinoma
    'Sarcoma': -0.0631,
    'Other': -0.3323,         # Default (same as Bladder)
}

# Mapping from AllData.xlsx CancerType values to LORIS calibration keys.
# AllData.xlsx uses descriptive names; LORIS uses abbreviations.
CANCER_TYPE_ALIAS = {
    'Head & Neck': 'HNSCC',
    'Renal': 'RCC',
    'Hepatobiliary': 'HCC',
    'Esophageal': 'Esophagogastric',
    'Gastric': 'Esophagogastric',
    'Endometrial': 'Other',
    'CNS': 'Glioma',
    'Unknown primary': 'CUP',
}


def preprocess_features(df, include_tmb=True):
    """
    Apply LORIS preprocessing: capping and imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw feature values.
    include_tmb : bool
        Whether to include TMB in output.

    Returns
    -------
    pd.DataFrame
        Preprocessed features.
    """
    result = pd.DataFrame(index=df.index)

    # TMB (capped at 50)
    if include_tmb and 'TMB' in df.columns:
        result['TMB'] = df['TMB'].clip(upper=TMB_CAP)

    # Systemic therapy history (0 or 1)
    if 'Systemic_therapy_history' in df.columns:
        result['PSTH'] = df['Systemic_therapy_history'].fillna(0).astype(int)

    # Albumin (impute missing with median)
    if 'Albumin' in df.columns:
        result['Albumin'] = df['Albumin'].fillna(ALBUMIN_IMPUTE)

    # NLR (capped at 25, impute missing)
    if 'NLR' in df.columns:
        result['NLR'] = df['NLR'].fillna(NLR_IMPUTE).clip(upper=NLR_CAP)

    # Age (capped at 85)
    if 'Age' in df.columns:
        result['Age'] = df['Age'].clip(upper=AGE_CAP)

    return result


def map_cancer_type(df):
    """
    Map cancer type information to LORIS calibration keys.

    Reads the ``CancerType`` string column (descriptive names like
    ``'Head & Neck'`` or ``'Renal'``), mapping each value to a LORIS key via
    :data:`CANCER_TYPE_ALIAS` and falling back to ``'Other'``.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a ``CancerType`` string column.

    Returns
    -------
    pd.Series
        LORIS calibration key per row (e.g. ``'NSCLC'``, ``'RCC'``).
    """
    # Map descriptive names → LORIS keys via alias, keep direct matches
    raw = df['CancerType'].copy()
    return raw.map(
        lambda ct: CANCER_TYPE_ALIAS.get(ct, ct)
        if ct in CANCER_TYPE_CALIBRATION or ct in CANCER_TYPE_ALIAS
        else 'Other'
    )


def calculate_loris(df):
    """
    Calculate LORIS scores using PUBLISHED coefficients.

    The LORIS score is the predicted probability of response to immunotherapy,
    calculated using logistic regression with published coefficients.

    Formula:
        logit(p) = intercept + TMB*coef_TMB + PSTH*coef_PSTH + Albumin*coef_Albumin
                   + NLR*coef_NLR + Age*coef_Age + cancer_type_calibration
        LORIS = 1 / (1 + exp(-logit(p)))

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns: TMB, Systemic_therapy_history, Albumin,
        NLR, Age, and CancerType.* indicators.

    Returns
    -------
    pd.Series
        LORIS scores (probability of response, 0-1).
    """
    # Preprocess features
    X = preprocess_features(df, include_tmb=True)

    # Get cancer type calibration terms
    cancer_types = map_cancer_type(df)
    calibration = cancer_types.map(CANCER_TYPE_CALIBRATION).fillna(CANCER_TYPE_CALIBRATION['Other'])

    # Calculate log-odds
    logit = (
        LORIS_COEFFICIENTS['intercept']
        + X['TMB'] * LORIS_COEFFICIENTS['TMB']
        + X['PSTH'] * LORIS_COEFFICIENTS['PSTH']
        + X['Albumin'] * LORIS_COEFFICIENTS['Albumin']
        + X['NLR'] * LORIS_COEFFICIENTS['NLR']
        + X['Age'] * LORIS_COEFFICIENTS['Age']
        + calibration
    )

    # Convert to probability
    loris = 1 / (1 + np.exp(-logit))

    return loris


def fit_lr5_no_tmb(X_train, y_train):
    """
    Fit LR5 (no TMB) model with StandardScaler (matching LORIS methodology).

    Since LR5 (no TMB) coefficients are not published, we train on Chowell data.
    Uses z-score standardization like LORIS, which is critical for proper
    regularization behavior.

    NOTE: The published LR5 (noTMB) achieved AUC=0.69 on Chowell test set
    (Supplementary Table 1).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (already preprocessed).
    y_train : np.ndarray
        Training labels (0/1 response).

    Returns
    -------
    tuple
        (LogisticRegression model, StandardScaler scaler)
    """
    # Z-score standardization (critical for proper regularization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Logistic regression with balanced class weights
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def calculate_lr5(df, model, scaler):
    """
    Calculate LR5 (no TMB) scores using fitted model and scaler.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features.
    model : LogisticRegression
        Fitted LR5 model.
    scaler : StandardScaler
        Fitted scaler for z-score normalization.

    Returns
    -------
    pd.Series
        LR5 scores (probability of response, 0-1).
    """
    X = preprocess_features(df, include_tmb=False)

    # Get cancer type indicators
    cancer_cols = ['CancerType.NSCLC', 'CancerType.Melanoma', 'CancerType.RCC']
    for col in cancer_cols:
        if col in df.columns:
            X[col] = df[col].fillna(0).astype(int)
        else:
            X[col] = 0

    # Ensure column order matches training
    feature_order = ['PSTH', 'Albumin', 'NLR', 'Age',
                     'CancerType.NSCLC', 'CancerType.Melanoma', 'CancerType.RCC']
    X = X[feature_order]

    # Apply same scaling as training
    X_scaled = scaler.transform(X)

    return pd.Series(model.predict_proba(X_scaled)[:, 1], index=df.index)


def print_loris_calculation_example(row, cancer_type):
    """Print step-by-step LORIS calculation for validation."""
    print("\n" + "="*60)
    print("LORIS Calculation Example (for web tool validation)")
    print("="*60)

    tmb = min(row['TMB'], TMB_CAP) if pd.notna(row['TMB']) else 0
    psth = row.get('Systemic_therapy_history', 0)
    albumin = row['Albumin'] if pd.notna(row['Albumin']) else ALBUMIN_IMPUTE
    nlr = min(row['NLR'] if pd.notna(row['NLR']) else NLR_IMPUTE, NLR_CAP)
    age = min(row['Age'], AGE_CAP)
    cal = CANCER_TYPE_CALIBRATION.get(cancer_type, CANCER_TYPE_CALIBRATION['Other'])

    print(f"\nInput values (after preprocessing):")
    print(f"  TMB:     {tmb:.1f} (capped at {TMB_CAP})")
    print(f"  PSTH:    {psth}")
    print(f"  Albumin: {albumin:.2f}")
    print(f"  NLR:     {nlr:.2f} (capped at {NLR_CAP})")
    print(f"  Age:     {age:.1f} (capped at {AGE_CAP})")
    print(f"  Cancer:  {cancer_type} (calibration: {cal:+.4f})")

    print(f"\nCoefficients:")
    print(f"  Intercept: {LORIS_COEFFICIENTS['intercept']:+.4f}")
    print(f"  TMB:       {LORIS_COEFFICIENTS['TMB']:+.4f}")
    print(f"  PSTH:      {LORIS_COEFFICIENTS['PSTH']:+.4f}")
    print(f"  Albumin:   {LORIS_COEFFICIENTS['Albumin']:+.4f}")
    print(f"  NLR:       {LORIS_COEFFICIENTS['NLR']:+.4f}")
    print(f"  Age:       {LORIS_COEFFICIENTS['Age']:+.4f}")

    # Calculate each term
    term_intercept = LORIS_COEFFICIENTS['intercept']
    term_tmb = tmb * LORIS_COEFFICIENTS['TMB']
    term_psth = psth * LORIS_COEFFICIENTS['PSTH']
    term_albumin = albumin * LORIS_COEFFICIENTS['Albumin']
    term_nlr = nlr * LORIS_COEFFICIENTS['NLR']
    term_age = age * LORIS_COEFFICIENTS['Age']

    print(f"\nCalculation:")
    print(f"  Intercept:      {term_intercept:+.4f}")
    print(f"  TMB term:       {tmb:.1f} × {LORIS_COEFFICIENTS['TMB']:.4f} = {term_tmb:+.4f}")
    print(f"  PSTH term:      {psth} × {LORIS_COEFFICIENTS['PSTH']:.4f} = {term_psth:+.4f}")
    print(f"  Albumin term:   {albumin:.2f} × {LORIS_COEFFICIENTS['Albumin']:.4f} = {term_albumin:+.4f}")
    print(f"  NLR term:       {nlr:.2f} × {LORIS_COEFFICIENTS['NLR']:.4f} = {term_nlr:+.4f}")
    print(f"  Age term:       {age:.1f} × {LORIS_COEFFICIENTS['Age']:.4f} = {term_age:+.4f}")
    print(f"  Cancer cal:     {cal:+.4f}")

    logit = term_intercept + term_tmb + term_psth + term_albumin + term_nlr + term_age + cal
    loris = 1 / (1 + np.exp(-logit))

    print(f"\n  Log-odds (sum): {logit:+.4f}")
    print(f"  LORIS score:    1 / (1 + exp({-logit:.4f})) = {loris:.4f}")
    print(f"\n  Compare with: https://loris.ccr.cancer.gov")


def main():
    """Main script to calculate LORIS and LR5 scores."""

    print("="*70)
    print("  LORIS and LR5 (no TMB) Score Calculator")
    print("  Based on Chang et al., Nature Cancer 2024")
    print("="*70)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    df_train = pd.read_csv(DATA_DIR / 'loris' / 'Chowell_train.csv')
    df_test = pd.read_csv(DATA_DIR / 'loris' / 'Chowell_test.csv')

    # The Chowell files key patients by SAMPLE_ID and carry a descriptive
    # CancerType string column. Rename the id and derive the three NSCLC /
    # Melanoma / RCC indicators the LR5 model expects from that string column.
    for df in (df_train, df_test):
        df.rename(columns={'SAMPLE_ID': 'PatientID'}, inplace=True)
        mapped = map_cancer_type(df)
        for key in ('NSCLC', 'Melanoma', 'RCC'):
            df[f'CancerType.{key}'] = (mapped == key).astype(int)

    print(f"  Train: {len(df_train)} samples, {df_train['Response'].sum()} responders")
    print(f"  Test:  {len(df_test)} samples, {df_test['Response'].sum()} responders")

    # -------------------------------------------------------------------------
    # Calculate LORIS scores (published coefficients)
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("LORIS Scores (Published Coefficients)")
    print("-"*70)

    df_train['LORIS'] = calculate_loris(df_train)
    df_test['LORIS'] = calculate_loris(df_test)

    # Evaluate on train/test
    for name, df in [("Train", df_train), ("Test", df_test)]:
        valid_mask = df['TMB'].notna()
        y_true = df.loc[valid_mask, 'Response']
        y_score = df.loc[valid_mask, 'LORIS']
        auc = roc_auc_score(y_true, y_score)
        print(f"\n{name} (n={valid_mask.sum()}, responders={y_true.sum()}):")
        print(f"  AUC: {auc:.3f}")
        print(f"  LORIS range: [{y_score.min():.3f}, {y_score.max():.3f}]")
        print(f"  LORIS mean:  {y_score.mean():.3f}")

    # -------------------------------------------------------------------------
    # Fit and evaluate LR5 (no TMB) model
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("LR5 (no TMB) Model - Training on Chowell Data")
    print("-"*70)
    print("\nUsing LORIS methodology:")
    print("  StandardScaler for z-score normalization")
    print("  LogisticRegression(C=1.0, class_weight='balanced')")

    # Prepare training data for LR5
    X_train_lr5 = preprocess_features(df_train, include_tmb=False)

    # Add cancer type indicators
    cancer_cols = ['CancerType.NSCLC', 'CancerType.Melanoma', 'CancerType.RCC']
    for col in cancer_cols:
        X_train_lr5[col] = df_train[col].fillna(0).astype(int)

    # Reorder columns
    feature_order = ['PSTH', 'Albumin', 'NLR', 'Age',
                     'CancerType.NSCLC', 'CancerType.Melanoma', 'CancerType.RCC']
    X_train_lr5 = X_train_lr5[feature_order]
    y_train = df_train['Response'].values

    # Fit LR5 model (returns model and scaler)
    lr5_model, lr5_scaler = fit_lr5_no_tmb(X_train_lr5, y_train)

    # Print fitted coefficients (on standardized features)
    print("\nFitted LR5 coefficients (standardized features):")
    print(f"  Intercept:            {lr5_model.intercept_[0]:+.4f}")
    for feat, coef in zip(feature_order, lr5_model.coef_[0]):
        print(f"  {feat:22s} {coef:+.4f}")

    # Calculate LR5 scores
    df_train['LR5_noTMB'] = calculate_lr5(df_train, lr5_model, lr5_scaler)
    df_test['LR5_noTMB'] = calculate_lr5(df_test, lr5_model, lr5_scaler)

    # Evaluate
    for name, df in [("Train", df_train), ("Test", df_test)]:
        y_true = df['Response']
        y_score = df['LR5_noTMB']
        auc = roc_auc_score(y_true, y_score)
        print(f"\n{name} (n={len(df)}, responders={y_true.sum()}):")
        print(f"  AUC: {auc:.3f}")
        print(f"  LR5 range: [{y_score.min():.3f}, {y_score.max():.3f}]")
        print(f"  LR5 mean:  {y_score.mean():.3f}")

    # -------------------------------------------------------------------------
    # Validation: Show detailed calculation for a few samples
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Validation Examples")
    print("-"*70)

    # Pick 3 diverse samples for validation
    sample_indices = [0, 50, 100]  # First, middle, later samples

    for idx in sample_indices:
        if idx < len(df_test):
            row = df_test.iloc[idx]
            cancer_type = map_cancer_type(df_test.iloc[[idx]]).values[0]
            print(f"\n--- Test Sample {idx} (PatientID: {row['PatientID']}) ---")
            print_loris_calculation_example(row, cancer_type)

    # -------------------------------------------------------------------------
    # Compare LORIS vs LR5 on test set
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Model Comparison on Test Set")
    print("-"*70)

    valid_mask = df_test['TMB'].notna()
    y_true_test = df_test.loc[valid_mask, 'Response']

    loris_auc = roc_auc_score(y_true_test, df_test.loc[valid_mask, 'LORIS'])
    lr5_auc = roc_auc_score(y_true_test, df_test.loc[valid_mask, 'LR5_noTMB'])

    print(f"\nOn {valid_mask.sum()} test samples with TMB:")
    print(f"  LORIS (with TMB) AUC: {loris_auc:.3f}")
    print(f"  LR5 (no TMB) AUC:     {lr5_auc:.3f}")
    print(f"  Difference:           {loris_auc - lr5_auc:+.3f}")

    # Correlation between scores
    corr = df_test.loc[valid_mask, 'LORIS'].corr(df_test.loc[valid_mask, 'LR5_noTMB'])
    print(f"\nCorrelation (LORIS vs LR5): {corr:.3f}")

    # -------------------------------------------------------------------------
    # Export scores
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Exporting Results")
    print("-"*70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Export test set scores
    output_cols = ['PatientID', 'Response', 'TMB', 'Age', 'Albumin', 'NLR',
                   'Systemic_therapy_history', 'LORIS', 'LR5_noTMB']

    # Add cancer type for reference
    df_test['CancerType'] = map_cancer_type(df_test)
    output_cols.append('CancerType')

    scores_df = df_test[output_cols].copy()
    scores_df.to_csv(OUTPUT_DIR / 'loris_scores.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'loris_scores.csv'}")

    # Export LR5 model coefficients and scaler parameters for reproducibility
    lr5_coef_df = pd.DataFrame({
        'Feature': ['Intercept'] + feature_order,
        'Coefficient': [lr5_model.intercept_[0]] + list(lr5_model.coef_[0]),
        'Scaler_Mean': [np.nan] + list(lr5_scaler.mean_),
        'Scaler_Scale': [np.nan] + list(lr5_scaler.scale_)
    })
    lr5_coef_df.to_csv(OUTPUT_DIR / 'lr5_coefficients.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'lr5_coefficients.csv'}")

    # Summary statistics
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"\nLORIS (published coefficients):")
    print(f"  - Uses 6 features: TMB, PSTH, Albumin, NLR, Age + cancer type")
    print(f"  - Test AUC: {loris_auc:.3f} (published: 0.75)")

    print(f"\nLR5 (trained on Chowell):")
    print(f"  - Uses 5 features: PSTH, Albumin, NLR, Age + cancer type indicators")
    print(f"  - Test AUC: {lr5_auc:.3f} (published: 0.69)")
    print(f"  - Coefficients saved to: {OUTPUT_DIR / 'lr5_coefficients.csv'}")

    print("\n" + "="*70)
    print("Sources")
    print("="*70)
    print("  - LORIS Paper: https://www.nature.com/articles/s43018-024-00772-7")
    print("  - PMC Full Text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11962634/")
    print("  - LORIS Web Tool: https://loris.ccr.cancer.gov")
    print("  - LORIS GitHub: https://github.com/rootchang/LORIS")


if __name__ == '__main__':
    main()
