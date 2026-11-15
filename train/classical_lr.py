"""Classical reference: TF-IDF + logistic regression (well-calibrated baseline, as in the BBS paper)."""
from __future__ import annotations

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import eval_sets, hyperparams, training_set


def _text(df):
    b = df.text_b.fillna("") if "text_b" in df else ""
    return (df.text_a.fillna("") + " [SEP] " + b).tolist()


def run(spec):
    hp = hyperparams(spec)
    t0 = time.time()
    tr = training_set(spec)
    vec = TfidfVectorizer(ngram_range=(1, hp["ngram_max"]), min_df=min(hp["min_df"], max(1, len(tr) // 50)),
                          max_features=hp["max_features"], sublinear_tf=True)
    X = vec.fit_transform(_text(tr))
    clf = LogisticRegression(C=hp["C"], max_iter=2000).fit(X, tr.label, sample_weight=tr.w / tr.w.mean())
    preds = {s: (df, clf.predict_proba(vec.transform(_text(df)))[:, 1]) for s, df in eval_sets(spec).items()}
    return preds, dict(train_seconds=round(time.time() - t0, 1), n_train=len(tr), hp=hp, gpu="cpu")
