"""The whole TabFM experiment as ONE script, for running on a Colab GPU runtime.

The notebook (tabfm_churn_demo.ipynb) is the version to READ and learn from.
This is the same experiment compressed into a single file, so it can be driven
over the Colab MCP server in two cells instead of thirty-eight.

Cell 1:  !pip install -q "tabfm[pytorch] @ git+https://github.com/google-research/tabfm" safetensors
Cell 2:  exec(open("colab_run.py").read())      # or just paste this file in

Prints one final results table. Nothing else needs reading.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- config
TABFM_SIZES      = [25, 50, 100, 250, 500, 1000, 2000]
TRADITIONAL_ONLY = [4000, 8000]
SEEDS            = [0, 1, 2]

CATEGORICAL = ["country", "gender", "plan_tier", "acquisition_channel"]
NUMERIC = ["age", "tenure_years", "balance", "num_products", "has_credit_card",
           "is_active_member", "estimated_salary", "support_tickets",
           "months_since_last_login"]


# ----------------------------------------------------------------- the data
def _draw(rng, n_rows):
    """The customer characteristics. Shared by the data and the oracle so the two
    stay in lockstep -- the oracle must see exactly the same customers."""
    d = {}
    d["country"] = rng.choice(["France", "Germany", "Spain"], n_rows, p=[0.5, 0.25, 0.25])
    d["gender"] = rng.choice(["Male", "Female"], n_rows)
    d["plan_tier"] = rng.choice(["Basic", "Plus", "Premium"], n_rows, p=[0.55, 0.3, 0.15])
    d["channel"] = rng.choice(["branch", "online", "partner", "referral"], n_rows,
                              p=[0.3, 0.4, 0.2, 0.1])
    d["age"] = np.clip(rng.normal(39, 11, n_rows), 18, 88).round(0)
    d["tenure"] = np.clip(rng.gamma(2.0, 2.2, n_rows), 0, 22).round(1)
    d["balance"] = np.where(rng.random(n_rows) < 0.28, 0.0,
                            np.abs(rng.normal(78_000, 45_000, n_rows))).round(2)
    d["num_products"] = rng.choice([1, 2, 3, 4], n_rows, p=[0.45, 0.4, 0.11, 0.04])
    d["has_cc"] = rng.binomial(1, 0.7, n_rows)
    d["active"] = rng.binomial(1, 0.52, n_rows)
    d["salary"] = np.clip(rng.normal(62_000, 26_000, n_rows), 12_000, None).round(2)
    d["tickets"] = rng.poisson(0.6, n_rows)
    d["idle"] = np.clip(rng.exponential(2.4, n_rows), 0, 36).round(1)
    return d


def _logit(d):
    """The hidden churn rule: thresholds and interactions, no noise."""
    z = -3.10
    z += 0.9 * ((d["age"] - 44) / 15.0) ** 2
    z += np.select([d["num_products"] == 1, d["num_products"] == 2,
                    d["num_products"] == 3, d["num_products"] >= 4],
                   [0.55, -0.75, 0.85, 1.7])
    z += 1.15 * (1 - d["active"])
    z += 0.62 * (d["country"] == "Germany")
    z += 0.45 * d["tickets"]
    z += 0.40 * (d["idle"] > 6)
    z -= 0.09 * np.minimum(d["tenure"], 8)
    z += 0.35 * (d["plan_tier"] == "Basic")
    z -= 0.30 * (d["plan_tier"] == "Premium")
    z += 0.85 * ((d["balance"] > 120_000) & (d["active"] == 0))
    z += 0.70 * ((d["channel"] == "partner") & (d["tenure"] < 2))
    return z


def make_churn_data(n_rows=12_000, seed=0):
    rng = np.random.default_rng(seed)
    d = _draw(rng, n_rows)
    z = _logit(d) + rng.normal(0, 0.55, n_rows)
    churn = rng.binomial(1, 1 / (1 + np.exp(-z)))

    df = pd.DataFrame({
        "age": d["age"], "tenure_years": d["tenure"], "balance": d["balance"],
        "num_products": d["num_products"], "has_credit_card": d["has_cc"],
        "is_active_member": d["active"], "estimated_salary": d["salary"],
        "support_tickets": d["tickets"], "months_since_last_login": d["idle"],
        "country": d["country"], "gender": d["gender"],
        "plan_tier": d["plan_tier"], "acquisition_channel": d["channel"],
    })
    df.loc[rng.random(n_rows) < 0.05, "balance"] = np.nan
    df.loc[rng.random(n_rows) < 0.03, "plan_tier"] = None
    return df, pd.Series(churn, name="churned")


def ceiling(y_test, n_rows=12_000, seed=0):
    """Best score achievable by anything: an oracle knowing the exact rule."""
    z = _logit(_draw(np.random.default_rng(seed), n_rows))
    noise = np.random.default_rng(7).normal(0, 0.55, (400, n_rows))
    p = (1 / (1 + np.exp(-(z + noise)))).mean(axis=0)[-len(y_test):]
    return roc_auc_score(y_test, p)


# ----------------------------------------------------------------- baselines
def build_traditional():
    prep = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    logistic = Pipeline([("p", prep), ("m", LogisticRegression(max_iter=2000))])

    boosting = Pipeline([
        ("p", ColumnTransformer([
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])),
        ("m", HistGradientBoostingClassifier(random_state=0)),
    ])
    return {"Gradient boosting": boosting, "Logistic regression": logistic}


# ----------------------------------------------------------------- tabfm
def load_tabfm():
    import inspect
    import torch
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as v1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tabfm] device={device}", flush=True)
    if device == "cpu":
        print("[tabfm] WARNING: no GPU. This will be very slow.", flush=True)

    try:
        model = v1.load(model_type="classification")
    except TypeError:
        model = v1.load()

    accepted = set(inspect.signature(TabFMClassifier.__init__).parameters)
    print(f"[tabfm] classifier accepts: {sorted(accepted - {'self'})}", flush=True)

    def predict(X_tr, y_tr, X_te):
        kwargs = {"model": model}
        if "device" in accepted:
            kwargs["device"] = device
        clf = TabFMClassifier(**kwargs)
        try:
            clf.fit(X_tr, y_tr.to_numpy())
        except (ValueError, TypeError):
            fill = lambda d: d.assign(**{c: d[c].fillna("unknown") for c in CATEGORICAL})
            clf = TabFMClassifier(**kwargs)
            clf.fit(fill(X_tr), y_tr.to_numpy())
            X_te = fill(X_te)
        return clf.predict_proba(X_te)[:, 1]

    return predict


# ----------------------------------------------------------------- main
def main():
    X_all, y_all = make_churn_data()
    X_test, y_test = X_all.iloc[-3000:], y_all.iloc[-3000:]
    X_pool, y_pool = X_all.iloc[:-3000], y_all.iloc[:-3000]

    CEIL = ceiling(y_test)
    print(f"[data] {len(X_all):,} customers, churn rate {y_all.mean():.1%}")
    print(f"[data] ceiling = {CEIL:.3f} ROC AUC (nothing can beat this)")
    print(flush=True)

    tabfm_predict = load_tabfm()
    print(flush=True)

    records = []
    for n in TABFM_SIZES + TRADITIONAL_ONLY:
        for seed in SEEDS:
            idx = np.random.default_rng(seed).choice(len(X_pool), n, replace=False)
            X_tr, y_tr = X_pool.iloc[idx], y_pool.iloc[idx]
            if y_tr.nunique() < 2:
                continue

            if n in TABFM_SIZES:
                t0 = time.perf_counter()
                auc = roc_auc_score(y_test, tabfm_predict(X_tr, y_tr, X_test))
                records.append({"n": n, "seed": seed, "method": "TabFM",
                                "auc": auc, "secs": time.perf_counter() - t0})

            for name, model in build_traditional().items():
                t0 = time.perf_counter()
                model.fit(X_tr, y_tr)
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                records.append({"n": n, "seed": seed, "method": name,
                                "auc": auc, "secs": time.perf_counter() - t0})

        row = pd.DataFrame(records).query("n == @n").groupby("method").auc.mean()
        print(f"[run] n={n:>5} | " + "  ".join(f"{m}={v:.3f}" for m, v in row.items()),
              flush=True)

    res = pd.DataFrame(records)
    auc = res.pivot_table(index="n", columns="method", values="auc").round(3)
    sec = res.pivot_table(index="n", columns="method", values="secs").round(2)

    print("\n" + "=" * 62)
    print(f"ROC AUC   (0.500 = useless, {CEIL:.3f} = ceiling)")
    print("=" * 62)
    print(auc.to_string())
    print("\n" + "=" * 62)
    print("SECONDS to fit + score 3,000 test customers")
    print("=" * 62)
    print(sec.to_string())

    res.to_csv("tabfm_results.csv", index=False)
    print("\nsaved: tabfm_results.csv")
    return auc, sec


if __name__ == "__main__":
    main()
