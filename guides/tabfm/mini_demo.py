"""A small, honest TabFM demonstration that finishes on free Colab hardware.

The full experiment (colab_run.py) is not runnable on a free T4: at TabFM's
default settings a single fit takes about five hours. This version keeps the
argument intact while fitting in roughly fifteen minutes.

What changed, and why:

  * TabFM is asked only about SMALL training sets (10 - 200 examples). That is
    exactly where its story lives, and where it is cheap, because the examples
    it must re-read for every prediction are few.
  * Gradient boosting and logistic regression run across the FULL range, up to
    8,000 rows. They cost hundredths of a second, so there is no reason not to.
  * The exam is 300 customers rather than 3,000, and TabFM uses ONE ensemble
    pass rather than its default of 32.

Those last two are real deviations from the default configuration and any number
produced here must be reported as such.

Every method sees identical training rows and is marked on the identical exam.
Results print as they are produced, so nothing is lost if the run is stopped.

    exec(open("mini_demo.py").read())
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import colab_run as cr  # importing does NOT run the big experiment

TABFM_SIZES = [10, 25, 50, 100, 200]
BASELINE_EXTRA = [500, 1000, 2000, 4000, 8000]  # traditional methods only
SEEDS = [0, 1, 2]
TEST_ROWS = 300
N_ESTIMATORS = 1
BATCH_SIZE = 16

# ------------------------------------------------------------------ the data
X, y = cr.make_churn_data()
X_test, y_test = X.iloc[-TEST_ROWS:], y.iloc[-TEST_ROWS:]
X_pool, y_pool = X.iloc[:-3000], y.iloc[:-3000]  # never overlaps the exam

CEIL = cr.ceiling(y_test)
print(f"[data] pool={len(X_pool):,} customers   exam={len(X_test)} customers   "
      f"churn in exam={y_test.mean():.1%}", flush=True)
print(f"[data] ceiling = {CEIL:.3f} ROC AUC on this exam (nothing can beat it)", flush=True)
print(f"[cfg ] TabFM: n_estimators={N_ESTIMATORS} (default 32), batch_size={BATCH_SIZE}",
      flush=True)
print(flush=True)

# ------------------------------------------------------------------ tabfm
import torch
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as v1

t0 = time.perf_counter()
tabfm_model = v1.load(model_type="classification")
print(f"[tabfm] loaded in {time.perf_counter() - t0:.0f}s  cuda={torch.cuda.is_available()}",
      flush=True)
print(flush=True)

fill = lambda d: d.assign(**{c: d[c].fillna("unknown") for c in cr.CATEGORICAL})
X_test_f = fill(X_test)

# ------------------------------------------------------------------ the loop
records = []

for n in TABFM_SIZES + BASELINE_EXTRA:
    for seed in SEEDS:
        idx = np.random.default_rng(seed).choice(len(X_pool), n, replace=False)
        X_tr, y_tr = X_pool.iloc[idx], y_pool.iloc[idx]
        if y_tr.nunique() < 2:
            print(f"[skip] n={n} seed={seed}: only one class in the sample", flush=True)
            continue

        if n in TABFM_SIZES:
            t0 = time.perf_counter()
            clf = TabFMClassifier(model=tabfm_model, batch_size=BATCH_SIZE,
                                  n_estimators=N_ESTIMATORS)
            clf.fit(fill(X_tr), y_tr.to_numpy())
            p = clf.predict_proba(X_test_f)[:, 1]
            records.append({"n": n, "seed": seed, "method": "TabFM",
                            "auc": roc_auc_score(y_test, p),
                            "secs": time.perf_counter() - t0})

        for name, model in cr.build_traditional().items():
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_test)[:, 1]
            records.append({"n": n, "seed": seed, "method": name,
                            "auc": roc_auc_score(y_test, p),
                            "secs": time.perf_counter() - t0})

    row = pd.DataFrame(records).query("n == @n").groupby("method").auc.mean()
    secs = pd.DataFrame(records).query("n == @n").groupby("method").secs.mean()
    bits = "  ".join(f"{m}={v:.3f}" for m, v in row.items())
    print(f"[run] n={n:>5}  {bits}   (TabFM {secs.get('TabFM', float('nan')):.0f}s/fit)",
          flush=True)

# ------------------------------------------------------------------ results
res = pd.DataFrame(records)
auc = res.pivot_table(index="n", columns="method", values="auc").round(3)

print("\n" + "=" * 66)
print(f"ROC AUC, averaged over {len(SEEDS)} draws "
      f"(0.500 = coin flip, {CEIL:.3f} = ceiling)")
print("=" * 66)
print(auc.to_string())

res.to_csv("mini_demo_results.csv", index=False)
print("\nsaved: mini_demo_results.csv")
print("DONE", flush=True)
