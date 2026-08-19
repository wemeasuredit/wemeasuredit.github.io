"""Extend the TabFM curve to 500 and 1,000 training rows, to find the crossing.

The demonstration (mini_demo.py) stopped TabFM at 200 training rows, which left
the most interesting question open: at what point does gradient boosting -- which
keeps improving with more data -- actually overtake it?

Settings are IDENTICAL to mini_demo.py so the new points sit on the same curve:
300-customer exam, n_estimators=1, batch_size=16, same held-out block.

Prints each fit the moment it finishes, so the run can be stopped at any point
without losing what has already been measured.

    exec(open("mini_extend.py").read())
"""

import time

import numpy as np
from sklearn.metrics import roc_auc_score

import colab_run as cr

SIZES = [500, 1000]
SEEDS = [0, 1, 2]
TEST_ROWS = 300          # identical to mini_demo.py
N_ESTIMATORS = 1         # identical to mini_demo.py
BATCH_SIZE = 16          # identical to mini_demo.py

X, y = cr.make_churn_data()
X_test, y_test = X.iloc[-TEST_ROWS:], y.iloc[-TEST_ROWS:]
X_pool, y_pool = X.iloc[:-3000], y.iloc[:-3000]

print(f"[data] exam={len(X_test)} customers, ceiling={cr.ceiling(y_test):.3f}", flush=True)
print("[ref ] from mini_demo: TabFM 0.721 @200 | boosting 0.720 @500, 0.729 @1000",
      flush=True)
print(flush=True)

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as v1

t0 = time.perf_counter()
tabfm_model = v1.load(model_type="classification")
print(f"[tabfm] loaded in {time.perf_counter() - t0:.0f}s", flush=True)
print(flush=True)

fill = lambda d: d.assign(**{c: d[c].fillna("unknown") for c in cr.CATEGORICAL})
X_test_f = fill(X_test)

results = {}
for n in SIZES:
    aucs = []
    for seed in SEEDS:
        idx = np.random.default_rng(seed).choice(len(X_pool), n, replace=False)
        X_tr, y_tr = X_pool.iloc[idx], y_pool.iloc[idx]

        t0 = time.perf_counter()
        clf = TabFMClassifier(model=tabfm_model, batch_size=BATCH_SIZE,
                              n_estimators=N_ESTIMATORS)
        clf.fit(fill(X_tr), y_tr.to_numpy())
        p = clf.predict_proba(X_test_f)[:, 1]
        auc = roc_auc_score(y_test, p)
        aucs.append(auc)

        print(f"[fit] n={n:>5} seed={seed}  TabFM auc={auc:.3f}  "
              f"({time.perf_counter() - t0:.0f}s)   running mean={np.mean(aucs):.3f}",
              flush=True)

    results[n] = float(np.mean(aucs))
    print(f"[run] n={n:>5}  TabFM={results[n]:.3f}  (mean of {len(aucs)} draws)\n",
          flush=True)

print("=" * 58)
for n, v in results.items():
    print(f"TabFM @ {n:>5} rows = {v:.3f}")
print("=" * 58)
print("DONE", flush=True)
