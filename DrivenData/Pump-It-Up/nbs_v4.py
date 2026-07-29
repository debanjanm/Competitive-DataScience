import logging
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.api.types import CategoricalDtype
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

warnings.simplefilter('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# recorded_by is constant across all rows -> zero signal
# wpt_name, num_private -> high-cardinality / near-empty, low signal
# coarse-to-fine duplicates -> keep one granularity per group
DROP_COLS = [
    'id', 'recorded_by', 'wpt_name', 'num_private',
    'extraction_type', 'extraction_type_group',   # keep extraction_type_class
    'quality_group',                               # keep water_quality
    'quantity_group',                               # keep quantity
    'source_type', 'source_class',                 # keep source
    'waterpoint_type_group',                        # keep waterpoint_type
    'payment_type',                                  # keep payment
]

# 0 is a placeholder for missing in these columns, not a real value
ZERO_AS_MISSING_COLS = ['longitude', 'latitude', 'construction_year', 'population', 'gps_height']

# stored as True/False, not strings -> must be numeric, not category dtype,
# or XGBoost's native categorical path crashes trying to arrow-encode bools
BOOL_COLS = ['public_meeting', 'permit']

N_FOLDS = 5
RANDOM_STATE = 42


def load_data():
    logger.info('loading raw csvs')
    train_values = pd.read_csv('./data/raw/Training set values.csv')
    train_labels = pd.read_csv('./data/raw/Training set labels.csv')
    test_values = pd.read_csv('./data/raw/Test set values.csv')

    # catch mislabeled/swapped raw files early with a clear message instead of
    # a cryptic KeyError several stages downstream
    if list(train_labels.columns) != ['id', 'status_group']:
        raise ValueError(
            f"Training set labels.csv has unexpected columns {list(train_labels.columns)}, "
            f"expected ['id', 'status_group'] — raw files may be mislabeled/swapped."
        )
    if train_values.shape[1] != test_values.shape[1]:
        raise ValueError(
            f"Training set values.csv has {train_values.shape[1]} cols but "
            f"Test set values.csv has {test_values.shape[1]} cols — raw files may be mislabeled/swapped."
        )

    train_raw = pd.merge(train_values, train_labels, on='id', how='left')
    logger.info('train_raw shape=%s test_values shape=%s', train_raw.shape, test_values.shape)
    return train_raw, test_values


def drop_unneeded_columns(train_df, test_df):
    logger.info('dropping unneeded columns')
    train_dropped = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
    test_dropped = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])
    logger.info('columns remaining: %d', train_dropped.shape[1])
    return train_dropped, test_dropped


def fix_placeholder_zeros(train_df, test_df):
    logger.info('replacing placeholder zeros with NaN in %s', ZERO_AS_MISSING_COLS)
    train_out = train_df.copy()
    test_out = test_df.copy()
    for col in ZERO_AS_MISSING_COLS:
        train_out[col] = train_out[col].replace(0, np.nan)
        test_out[col] = test_out[col].replace(0, np.nan)
    logger.info('missing rate after zero-fix:\n%s', train_out[ZERO_AS_MISSING_COLS].isnull().mean())
    return train_out, test_out


def cast_boolean_columns(train_df, test_df):
    logger.info('casting boolean columns to 0/1 float: %s', BOOL_COLS)
    train_out = train_df.copy()
    test_out = test_df.copy()
    for col in BOOL_COLS:
        train_out[col] = train_out[col].map({True: 1.0, False: 0.0})
        test_out[col] = test_out[col].map({True: 1.0, False: 0.0})
    return train_out, test_out


def add_missingness_flags(train_df, test_df):
    """Whether a value was missing can itself be predictive (e.g. no scheme_name
    recorded often correlates with poorly-managed pumps) -> capture it before imputing."""
    cols_with_na = [c for c in train_df.columns if c != 'status_group' and train_df[c].isnull().any()]
    logger.info('adding missingness flags for: %s', cols_with_na)

    train_out = train_df.copy()
    test_out = test_df.copy()
    for col in cols_with_na:
        train_out[f'{col}_missing'] = train_out[col].isnull()
        test_out[f'{col}_missing'] = test_out[col].isnull()
    return train_out, test_out


def geo_impute_numeric(train_df, test_df, cols):
    """Fill missing numeric values using the train-set median for that column
    within the same region, falling back to the train-set global median for
    regions with no data (or unseen in test)."""
    logger.info('geo-imputing numeric cols by region median: %s', cols)
    train_out = train_df.copy()
    test_out = test_df.copy()

    region_medians = train_out.groupby('region')[cols].median()
    global_medians = train_out[cols].median()

    for col in cols:
        for df in (train_out, test_out):
            fill_by_region = df['region'].map(region_medians[col])
            df[col] = df[col].fillna(fill_by_region)
            df[col] = df[col].fillna(global_medians[col])

    return train_out, test_out


def add_pump_age(df):
    df = df.copy()
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['pump_age'] = df['date_recorded'].dt.year - df['construction_year']
    df['record_year'] = df['date_recorded'].dt.year
    df['record_month'] = df['date_recorded'].dt.month
    return df.drop(columns=['date_recorded'])


def engineer_features(train_df, test_df):
    logger.info('engineering pump_age / record_year / record_month')
    return add_pump_age(train_df), add_pump_age(test_df)


def impute_categorical(train_df, test_df):
    logger.info('imputing categorical cols with most-frequent value')
    categorical_cols = [c for c in train_df.select_dtypes(include=['object']).columns if c != 'status_group']

    cat_imputer = SimpleImputer(strategy='most_frequent')
    train_out = train_df.copy()
    test_out = test_df.copy()

    train_out[categorical_cols] = cat_imputer.fit_transform(train_out[categorical_cols])
    test_out[categorical_cols] = cat_imputer.transform(test_out[categorical_cols])
    return train_out, test_out, categorical_cols


def numeric_safety_net(train_df, test_df):
    """Any numeric column still holding NaN after geo-impute (shouldn't be many)
    gets a plain median fallback so training never chokes on leftover NaNs."""
    continuous_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c != 'status_group']
    num_imputer = SimpleImputer(strategy='median')

    train_out = train_df.copy()
    test_out = test_df.copy()
    train_out[continuous_cols] = num_imputer.fit_transform(train_out[continuous_cols])
    test_out[continuous_cols] = num_imputer.transform(test_out[continuous_cols])

    assert train_out.drop(columns=['status_group']).isnull().sum().sum() == 0
    assert test_out.isnull().sum().sum() == 0
    logger.info('no missing values left')
    return train_out, test_out, continuous_cols


def cast_categoricals(train_df, test_df, categorical_cols):
    """XGBoost's native categorical support (enable_categorical) needs pandas
    'category' dtype -> avoids frequency-encoding's information loss.
    Unseen test categories become NaN, which XGBoost treats like any other
    missing value at split time."""
    logger.info('casting categorical cols to pandas category dtype')
    train_out = train_df.copy()
    test_out = test_df.copy()
    for col in categorical_cols:
        cat_type = CategoricalDtype(categories=train_out[col].dropna().unique())
        train_out[col] = train_out[col].astype(cat_type)
        test_out[col] = test_out[col].astype(cat_type)
    return train_out, test_out


def log_mutual_information(train_df, feature_cols):
    """Informational only -- with native categorical + tree model we keep all
    features rather than truncating to a top-N MI cutoff."""
    le = LabelEncoder()
    y = le.fit_transform(train_df['status_group'])

    mi_input = train_df[feature_cols].copy()
    for col in mi_input.select_dtypes(include=['category']).columns:
        mi_input[col] = mi_input[col].cat.codes

    mi_scores = mutual_info_classif(mi_input, y, random_state=RANDOM_STATE)
    mi_ranked = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
    logger.info('mutual information ranking:\n%s', mi_ranked)
    return le, y


def cross_validate_and_ensemble(X, y, X_submit, le):
    logger.info('running %d-fold stratified CV', N_FOLDS)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_accuracies = []
    test_proba_sum = np.zeros((len(X_submit), len(le.classes_)))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        sample_weight = compute_sample_weight('balanced', y_train)

        clf = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            enable_categorical=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)

        val_preds = clf.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        fold_accuracies.append(acc)
        logger.info('fold %d/%d accuracy: %.4f', fold, N_FOLDS, acc)

        if fold == N_FOLDS:
            logger.info('fold %d confusion matrix:\n%s', fold, confusion_matrix(y_val, val_preds))
            logger.info('fold %d classification report:\n%s', fold,
                        classification_report(y_val, val_preds, target_names=le.classes_))

        test_proba_sum += clf.predict_proba(X_submit)

    logger.info('CV accuracy: %.4f +/- %.4f', np.mean(fold_accuracies), np.std(fold_accuracies))

    test_proba_avg = test_proba_sum / N_FOLDS
    test_preds = np.argmax(test_proba_avg, axis=1)
    return test_preds


def make_submission(test_preds, test_values, le, out_path='submission_v4.csv'):
    submission_df = pd.DataFrame({
        'id': test_values['id'],
        'status_group': le.inverse_transform(test_preds),
    })
    submission_df.to_csv(out_path, index=False)
    logger.info('submission written to %s (%d rows)', out_path, len(submission_df))
    return submission_df


def main():
    train_raw, test_values = load_data()
    train_dropped, test_dropped = drop_unneeded_columns(train_raw, test_values)
    train_zerofix, test_zerofix = fix_placeholder_zeros(train_dropped, test_dropped)
    train_boolfix, test_boolfix = cast_boolean_columns(train_zerofix, test_zerofix)
    train_flagged, test_flagged = add_missingness_flags(train_boolfix, test_boolfix)
    train_geo, test_geo = geo_impute_numeric(train_flagged, test_flagged, ZERO_AS_MISSING_COLS)
    train_featured, test_featured = engineer_features(train_geo, test_geo)
    train_catimp, test_catimp, categorical_cols = impute_categorical(train_featured, test_featured)
    train_clean, test_clean, continuous_cols = numeric_safety_net(train_catimp, test_catimp)
    train_final, test_final = cast_categoricals(train_clean, test_clean, categorical_cols)

    feature_cols = [c for c in train_final.columns if c != 'status_group']
    le, y = log_mutual_information(train_final, feature_cols)

    X = train_final[feature_cols]
    X_submit = test_final[feature_cols]

    test_preds = cross_validate_and_ensemble(X, y, X_submit, le)
    make_submission(test_preds, test_values, le)
    logger.info('pipeline finished')


if __name__ == '__main__':
    main()
