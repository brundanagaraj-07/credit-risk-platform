"""
Rule Derivation Module
-----------------------
Converts the opaque LightGBM model into a small set of business-readable
IF/THEN credit policy rules that a risk analyst (not a data scientist) can
read, challenge and put in front of an auditor.

Approach (documented in README "Rule derivation logic"):
  1. Fit a shallow (depth<=4) surrogate Decision Tree on the SAME features
     the LightGBM model uses, trained to mimic the LightGBM model's
     predicted probabilities (not the raw labels) -- this is a standard
     "model distillation" technique for explainability.
  2. Walk every root-to-leaf path in the surrogate tree and turn it into a
     natural-language rule, tagged with the leaf's average predicted risk
     and the number of training rows it covers (its "support").
  3. Keep only rules with reasonable support (>0.5% of data) so the output
     is a handful of decision-relevant policies, not overfit micro-rules.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, _tree


def _tree_to_rules(tree, feature_names, class_names=None):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], path + [f"{name} <= {threshold:.2f}"])
            recurse(tree_.children_right[node], path + [f"{name} > {threshold:.2f}"])
        else:
            value = float(tree_.value[node][0][0])
            support = int(tree_.n_node_samples[node])
            rules.append({"conditions": path, "predicted_risk": value, "support": support})

    recurse(0, [])
    return rules


def derive_business_rules(model, X: pd.DataFrame, y: pd.Series, max_depth: int = 4, top_k: int = 12) -> list[dict]:
    model_scores = model.predict_proba(X)[:, 1]

    surrogate = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=max(50, int(0.005 * len(X))))
    surrogate.fit(X, model_scores)

    raw_rules = _tree_to_rules(surrogate, list(X.columns))
    min_support = max(20, int(0.005 * len(X)))
    rules = [r for r in raw_rules if r["support"] >= min_support]
    rules.sort(key=lambda r: r["predicted_risk"], reverse=True)

    formatted = []
    for r in rules[:top_k]:
        band = "High" if r["predicted_risk"] >= 0.5 else ("Medium" if r["predicted_risk"] >= 0.2 else "Low")
        formatted.append({
            "rule": "IF " + " AND ".join(r["conditions"]) + f" THEN risk band = {band}",
            "predicted_default_rate": round(r["predicted_risk"], 3),
            "support_rows": r["support"],
            "support_pct": round(100 * r["support"] / len(X), 2),
            "risk_band": band,
        })
    return formatted
