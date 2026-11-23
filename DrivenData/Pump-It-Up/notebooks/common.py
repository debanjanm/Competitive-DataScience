"""Shared helpers for the Pump-It-Up notebooks: load a version's pipeline code,
fixed CV folds (identical for every notebook), and an append-only results log.

Notebooks own the *experiment*; implementation stays in vN/pipelines.py.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parent
RESULTS = NB_DIR / 'results.jsonl'
SEED = 42
N_FOLDS = 5
TARGET = 'status_group'


def load_version(v):
    """Import vN/pipelines.py under a unique module name (every version calls it `pipelines`)."""
    name = f'pipelines_{v}'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / v / 'pipelines.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_train(v='v2'):
    """Cleaned training frame (structural cleaning only, nothing learned) + encoded target."""
    m = load_version(v)
    df = m.DataPipeline().clean(m.DataPipeline().load_train())
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    return df.reset_index(drop=True), y, le


def folds(y):
    """THE validation scheme: stratified 5-fold, seed 42. Same split in every notebook,
    so differences between rows of a results table come from the change, not the split."""
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def log_result(version, label, scores, note=''):
    """Append (replace if same version+label) one row to results.jsonl."""
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines()] if RESULTS.exists() else []
    rows = [r for r in rows if (r['version'], r['label']) != (version, label)]
    rows.append({'version': version, 'label': label, 'metric': 'accuracy',
                 'mean': float(np.mean(scores)), 'std': float(np.std(scores)),
                 'folds': [float(s) for s in scores], 'note': note})
    RESULTS.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    return rows[-1]


def cv_xgb(df, y, make_features, params, balanced=False, y_to_features=False):
    """5-fold CV on the shared folds. Features are fit inside each fold (nothing
    learned from the validation rows). Returns (per-fold accuracy, OOF predictions)."""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight

    X_all = df.drop(columns=[TARGET])
    oof, scores = np.zeros(len(y), dtype=int), []
    for tr, va in folds(y):
        fp = make_features()
        Xtr = fp.fit_transform(X_all.iloc[tr], y[tr]) if y_to_features else fp.fit_transform(X_all.iloc[tr])
        Xva = fp.transform(X_all.iloc[va])
        clf = xgb.XGBClassifier(**params, tree_method='hist', enable_categorical=True,
                                random_state=SEED, n_jobs=-1)
        clf.fit(Xtr, y[tr], sample_weight=compute_sample_weight('balanced', y[tr]) if balanced else None)
        oof[va] = clf.predict(Xva)
        scores.append(accuracy_score(y[va], oof[va]))
    return scores, oof


def make_flex(version='v2'):
    """Version's FeaturePipeline with each v2-era step switchable (v2 and v3 share identical pipelines.py)."""
    base = load_version(version).FeaturePipeline

    class Flex(base):
        def __init__(self, geo=True, flags=True, fix_lat=False, k=0, text=None, drop=(), inter=()):
            self.geo, self.flags, self.fix_lat, self.k = geo, flags, fix_lat, k   # k>0: add KMeans geo_cluster (v4)
            # text=dict(cols=[...], min=N): lowercase+strip those columns, collapse levels seen < N times
            # in the training split to 'other' (v5); drop=cols removed outright
            self.text, self.drop = text, tuple(drop)
            self.inter = tuple(inter)   # subset of {'qw','pm','pop'}: v6 interaction features

        def _clean_text_flex(self, df, fit):
            df = df.copy()
            if fit:
                self.text_keep_ = {}
            for col in self.text['cols']:
                df[col] = df[col].str.lower().str.strip()
                if self.text.get('min', 0):
                    if fit:
                        counts = df[col].value_counts()
                        self.text_keep_[col] = set(counts[counts >= self.text['min']].index)
                    df[col] = df[col].where(df[col].isnull() | df[col].isin(self.text_keep_[col]), 'other')
            return df

        def _run(self, df, fit):
            if self.text:
                df = self._clean_text_flex(df, fit)
            if self.drop:
                df = df.drop(columns=list(self.drop))
            if self.fix_lat:
                df = df.copy()
                df['latitude'] = df['latitude'].where(df['latitude'] < -1e-4)
            if self.flags:
                df = self._add_missingness_flags(df, fit)
            elif fit:
                self.flag_cols_ = []
            if self.geo:
                df = self._impute_geo_numeric(df, fit)
            if self.k:
                from sklearn.cluster import KMeans
                df = df.copy()
                if fit:
                    self.kmeans_ = KMeans(n_clusters=self.k, random_state=SEED, n_init=10).fit(df[['latitude', 'longitude']])
                df['geo_cluster'] = self.kmeans_.predict(df[['latitude', 'longitude']]).astype(str)
            if self.inter:
                df = self._interactions_flex(df, fit)
            df = self._add_pump_age(df)
            df = self._impute_categorical(df, fit)
            df = self._impute_numeric(df, fit)
            df = self._cast_categoricals(df, fit)
            if fit:
                self.feature_cols_ = list(df.columns)
            return df[self.feature_cols_]

        def _interactions_flex(self, df, fit):
            df = df.copy()
            cat = {'qw': ('quantity', 'waterpoint_type', 'quantity_x_waterpoint_type'),
                   'pm': ('payment', 'management', 'payment_x_management')}
            for key, (a, b, new) in cat.items():
                if key in self.inter:
                    df[new] = df[a].astype(str) + '__' + df[b].astype(str)
            if 'pop' in self.inter:
                if fit:
                    self.region_pop_mean_ = df.groupby('region')['population'].mean()
                    self.global_pop_mean_ = df['population'].mean()
                df['population_density_ratio'] = (
                    df['population'] / df['region'].map(self.region_pop_mean_).fillna(self.global_pop_mean_))
            return df

    return Flex


def cv_xgb_es(df, y, make_features, params, balanced=False, es='none',
              max_estimators=2000, rounds=30):
    """Like cv_xgb, with early-stopping modes:
       'none'  fixed n_estimators from params (default 300)
       'val'   stop on the validation fold itself (what v3 shipped: the score is optimistic)
       'inner' stop on a 10% holdout carved from the TRAINING fold (honest)
    Returns (per-fold accuracy, OOF predictions, best iteration per fold)."""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_sample_weight

    X_all = df.drop(columns=[TARGET])
    oof, scores, iters = np.zeros(len(y), dtype=int), [], []
    for tr, va in folds(y):
        fp = make_features()
        Xtr, Xva = fp.fit_transform(X_all.iloc[tr]), fp.transform(X_all.iloc[va])
        ytr = y[tr]
        kw = dict(params)
        if es == 'none':
            kw.setdefault('n_estimators', 300)
            fit_X, fit_y, eval_set = Xtr, ytr, None
        else:
            kw.pop('n_estimators', None)
            kw.update(n_estimators=max_estimators, early_stopping_rounds=rounds, eval_metric='mlogloss')
            if es == 'val':
                fit_X, fit_y, eval_set = Xtr, ytr, [(Xva, y[va])]
            else:
                a, b = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=SEED)
                fit_X, fit_y, eval_set = Xtr.iloc[a], ytr[a], [(Xtr.iloc[b], ytr[b])]
        clf = xgb.XGBClassifier(**kw, tree_method='hist', enable_categorical=True,
                                random_state=SEED, n_jobs=-1)
        clf.fit(fit_X, fit_y, sample_weight=compute_sample_weight('balanced', fit_y) if balanced else None,
                eval_set=eval_set, verbose=False)
        oof[va] = clf.predict(Xva)
        scores.append(accuracy_score(y[va], oof[va]))
        iters.append(int(getattr(clf, 'best_iteration', kw.get('n_estimators', 0)) or 0) + 1 if es != 'none' else kw['n_estimators'])
    return scores, oof, iters
