"""Direct pycox vs mlsurv DeepHit comparison.

Isolates the early stopping fix by using identical architecture and
hyperparameters for both raw pycox and the mlsurv wrapper. No tuning.

Usage:
    conda run -n mlsurv python tests/benchmarks/bench_deephit_pycox_ref.py
    conda run -n mlsurv python tests/benchmarks/bench_deephit_pycox_ref.py --dataset METABRIC
"""

import argparse
import time
import warnings

import numpy as np
import torch
import torchtuples as tt
from pycox.models import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

from mlsurv.models import get_model
from mlsurv.utils import to_structured_array


def load_pycox_dataset(name):
    """Load a pycox dataset."""
    from pycox.datasets import flchain, metabric, support

    loaders = {"SUPPORT": support, "METABRIC": metabric, "FLCHAIN": flchain}
    loader = loaders[name]

    if name == "FLCHAIN":
        df = loader.read_df(processed=False)
        df = (
            df.drop(["chapter", "rownames"], axis=1, errors="ignore")
            .loc[lambda x: x["creatinine"].notna()]
            .reset_index(drop=True)
            .assign(sex=lambda x: (x["sex"] == "M").astype(float))
            .rename(columns={"futime": "duration", "death": "event"})
        )
        for col in df.columns:
            if col not in ["duration", "event"]:
                df[col] = df[col].astype("float32")
    else:
        # SUPPORT's x1-x6 are nominal categoricals that Kvamme one-hot encodes
        # only for the classical Cox baseline. This reference comparison targets
        # the embedding-free neural nets, which are tuned on the raw integer
        # codes, so we keep them as-is (one-hot is reserved for the Cox path;
        # see examples/vignette_03_benchmarks.ipynb).
        df = loader.read_df()

    df = df[df["duration"] > 0]
    return df


def run_pycox_raw(X_train, y_train, X_test, y_test, seed, hparams):
    """Train raw pycox DeepHit with proper early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split train into train/val for early stopping
    idx = np.arange(len(X_train))
    idx_tr, idx_val = train_test_split(
        idx, test_size=0.2, stratify=y_train["event"], random_state=seed
    )

    X_tr = X_train.values[idx_tr].astype("float32")
    X_val = X_train.values[idx_val].astype("float32")
    X_te = X_test.values.astype("float32")

    labtrans = LabTransDiscreteTime(cuts=hparams["num_durations"])
    y_tr = labtrans.fit_transform(
        y_train["duration"].values[idx_tr], y_train["event"].values[idx_tr]
    )
    y_val = labtrans.transform(
        y_train["duration"].values[idx_val], y_train["event"].values[idx_val]
    )

    net = tt.practical.MLPVanilla(
        X_tr.shape[1],
        hparams["layers"],
        labtrans.out_features,
        batch_norm=True,
        dropout=hparams["dropout"],
    )
    model = DeepHitSingle(
        net,
        tt.optim.Adam(lr=hparams["lr"]),
        alpha=hparams["alpha"],
        sigma=hparams["sigma"],
        duration_index=labtrans.cuts,
    )

    callbacks = [tt.callbacks.EarlyStopping(patience=hparams["patience"])]
    val_data = (
        torch.tensor(X_val),
        (torch.tensor(y_val[0]), torch.tensor(y_val[1])),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            torch.tensor(X_tr),
            (torch.tensor(y_tr[0]), torch.tensor(y_tr[1])),
            batch_size=hparams["batch_size"],
            epochs=hparams["epochs"],
            callbacks=callbacks,
            verbose=False,
            val_data=val_data,
        )

    epochs_trained = len(model.log.to_pandas())

    # Predict
    with torch.no_grad():
        surv = model.predict_surv_df(torch.tensor(X_te))
    median_times = []
    for col in surv.columns:
        idx_below = np.where(surv[col].values < 0.5)[0]
        if len(idx_below) > 0:
            median_times.append(surv.index[idx_below[0]])
        else:
            median_times.append(surv.index[-1])
    risk = -np.array(median_times)

    c_index = concordance_index_censored(
        y_test["event"].values.astype(bool),
        y_test["duration"].values,
        risk,
    )[0]
    return c_index, epochs_trained


def run_mlsurv_wrapper(X_train, y_train_sa, X_test, y_test_sa, seed, hparams):
    """Train mlsurv DeepHitModel wrapper."""
    model = get_model(
        "deephit",
        layers=hparams["layers"],
        dropout=hparams["dropout"],
        alpha=hparams["alpha"],
        sigma=hparams["sigma"],
        learning_rate=hparams["lr"],
        batch_size=hparams["batch_size"],
        epochs=hparams["epochs"],
        num_durations=hparams["num_durations"],
        patience=hparams["patience"],
        val_fraction=0.2,
        random_state=seed,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train_sa)

    epochs_trained = len(model.model_.log.to_pandas())
    risk = model.predict(X_test)

    c_index = concordance_index_censored(
        y_test_sa["event"], y_test_sa["time"], risk
    )[0]
    return c_index, epochs_trained


def main():
    parser = argparse.ArgumentParser(
        description="Direct pycox vs mlsurv DeepHit comparison"
    )
    parser.add_argument(
        "--dataset",
        choices=["SUPPORT", "METABRIC", "FLCHAIN"],
        help="Single dataset (default: all three)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["SUPPORT", "METABRIC", "FLCHAIN"]

    # Shared hyperparameters (reasonable defaults, no tuning)
    # alpha=0.2 here intentionally matches pycox's library default so the
    # raw-pycox vs mlsurv-wrapper comparison is apples-to-apples. mlsurv's
    # own default differs (alpha=0.0).
    hparams = {
        "layers": [64, 64],
        "dropout": 0.1,
        "alpha": 0.2,
        "sigma": 0.1,
        "lr": 0.01,
        "batch_size": 256,
        "epochs": 200,
        "num_durations": 100,
        "patience": 10,
    }

    print("DeepHit: pycox raw vs mlsurv wrapper (identical hyperparameters)")
    print(f"  Architecture: {hparams['layers']}, dropout={hparams['dropout']}")
    print(f"  Training: lr={hparams['lr']}, batch={hparams['batch_size']}, "
          f"max_epochs={hparams['epochs']}, patience={hparams['patience']}")
    print(f"  Seed: {args.seed}")

    results = []

    for ds_name in datasets:
        print(f"\n{'=' * 65}")
        print(f"  {ds_name}")
        print(f"{'=' * 65}")

        df = load_pycox_dataset(ds_name)
        X = df.drop(columns=["duration", "event"])
        y_df = df[["duration", "event"]].copy()
        print(f"  n={len(df)}, features={X.shape[1]}, "
              f"events={int(y_df['event'].sum())}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_df, test_size=0.2, stratify=y_df["event"],
            random_state=args.seed,
        )

        # Scale
        scaler = StandardScaler()
        X_train_sc = X_train.copy()
        X_test_sc = X_test.copy()
        X_train_sc[:] = scaler.fit_transform(X_train)
        X_test_sc[:] = scaler.transform(X_test)

        # SurvivalArrays for mlsurv
        y_train_sa = to_structured_array(
            y_train["duration"].values, y_train["event"].values
        )
        y_test_sa = to_structured_array(
            y_test["duration"].values, y_test["event"].values
        )

        # Run pycox raw
        t0 = time.time()
        c_pycox, ep_pycox = run_pycox_raw(
            X_train_sc, y_train, X_test_sc, y_test, args.seed, hparams
        )
        t_pycox = time.time() - t0

        # Run mlsurv wrapper
        t0 = time.time()
        c_mlsurv, ep_mlsurv = run_mlsurv_wrapper(
            X_train_sc, y_train_sa, X_test_sc, y_test_sa, args.seed, hparams
        )
        t_mlsurv = time.time() - t0

        diff = c_mlsurv - c_pycox

        results.append({
            "dataset": ds_name,
            "c_pycox": c_pycox,
            "ep_pycox": ep_pycox,
            "c_mlsurv": c_mlsurv,
            "ep_mlsurv": ep_mlsurv,
            "diff": diff,
        })

        print(f"  pycox raw:     C={c_pycox:.4f}  ({ep_pycox} epochs, {t_pycox:.1f}s)")
        print(f"  mlsurv wrap:   C={c_mlsurv:.4f}  ({ep_mlsurv} epochs, {t_mlsurv:.1f}s)")
        print(f"  diff (wrap-raw): {diff:+.4f}")

    # Summary
    print(f"\n{'=' * 65}")
    print("  Summary: pycox raw vs mlsurv wrapper")
    print(f"{'=' * 65}")
    print(f"  {'Dataset':<10} {'pycox':>7} {'ep':>4}  {'mlsurv':>7} {'ep':>4}  {'diff':>7}")
    print(f"  {'-' * 48}")
    for r in results:
        print(
            f"  {r['dataset']:<10} {r['c_pycox']:>7.4f} {r['ep_pycox']:>4}  "
            f"{r['c_mlsurv']:>7.4f} {r['ep_mlsurv']:>4}  {r['diff']:>+7.4f}"
        )
    diffs = [abs(r["diff"]) for r in results]
    print(f"  {'-' * 48}")
    print(f"  Mean |diff|: {np.mean(diffs):.4f}")
    print(f"  Max  |diff|: {np.max(diffs):.4f}")


if __name__ == "__main__":
    main()
