import logging
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

TOP_N_FEATURES = 15


class FrequencyEncoder:
    """Maps each category to its train-set frequency; unseen test categories fall back to 0."""

    def __init__(self, cols):
        self.cols = cols
        self.freq_maps = {}

    def fit(self, X):
        for col in self.cols:
            self.freq_maps[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.freq_maps[col]).fillna(0)
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


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


def impute_missing_values(train_df, test_df):
    logger.info('imputing missing values')
    continuous_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c != 'status_group']
    categorical_cols = [c for c in train_df.select_dtypes(include=['object']).columns if c != 'status_group']

    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    train_out = train_df.copy()
    test_out = test_df.copy()

    train_out[continuous_cols] = num_imputer.fit_transform(train_out[continuous_cols])
    test_out[continuous_cols] = num_imputer.transform(test_out[continuous_cols])

    train_out[categorical_cols] = cat_imputer.fit_transform(train_out[categorical_cols])
    test_out[categorical_cols] = cat_imputer.transform(test_out[categorical_cols])

    assert train_out.drop(columns=['status_group']).isnull().sum().sum() == 0
    assert test_out.isnull().sum().sum() == 0
    logger.info('no missing values left')
    return train_out, test_out, continuous_cols, categorical_cols


def encode_and_scale(train_df, test_df, continuous_cols, categorical_cols):
    logger.info('scaling continuous cols + frequency-encoding categorical cols')
    scaler = StandardScaler()
    train_out = train_df.copy()
    test_out = test_df.copy()

    train_out[continuous_cols] = scaler.fit_transform(train_out[continuous_cols])
    test_out[continuous_cols] = scaler.transform(test_out[continuous_cols])

    freq_encoder = FrequencyEncoder(categorical_cols)
    train_out[categorical_cols] = freq_encoder.fit_transform(train_out[categorical_cols])
    test_out[categorical_cols] = freq_encoder.transform(test_out[categorical_cols])

    feature_cols = continuous_cols + categorical_cols
    return train_out, test_out, feature_cols


def select_features(train_df, feature_cols):
    logger.info('selecting top %d features by mutual information', TOP_N_FEATURES)
    le = LabelEncoder()
    y = le.fit_transform(train_df['status_group'])
    X = train_df[feature_cols]

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_ranked = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
    logger.info('mutual information ranking:\n%s', mi_ranked)

    top_features = mi_ranked.head(TOP_N_FEATURES).index.tolist()
    logger.info('top features: %s', top_features)
    return X[top_features], y, le, top_features


def train_model(X, y):
    logger.info('training XGBClassifier')
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    logger.info('training complete')
    return clf, X_test, y_test


def evaluate(clf, X_test, y_test, le):
    logger.info('evaluating on holdout split')
    val_preds = clf.predict(X_test)
    logger.info('accuracy: %.4f', accuracy_score(y_test, val_preds))
    logger.info('confusion matrix:\n%s', confusion_matrix(y_test, val_preds))
    logger.info('classification report:\n%s', classification_report(y_test, val_preds, target_names=le.classes_))


def make_submission(clf, X, y, X_submit, test_values, le, out_path='submission_v3.csv'):
    logger.info('refitting on full train set before predicting real test set')
    clf.fit(X, y)
    test_preds = clf.predict(X_submit)

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
    train_featured, test_featured = engineer_features(train_zerofix, test_zerofix)
    train_imputed, test_imputed, continuous_cols, categorical_cols = impute_missing_values(train_featured, test_featured)
    train_encoded, test_encoded, feature_cols = encode_and_scale(train_imputed, test_imputed, continuous_cols, categorical_cols)

    X, y, le, top_features = select_features(train_encoded, feature_cols)
    X_submit = test_encoded[top_features]

    clf, X_test, y_test = train_model(X, y)
    evaluate(clf, X_test, y_test, le)

    make_submission(clf, X, y, X_submit, test_values, le)
    logger.info('pipeline finished')


if __name__ == '__main__':
    main()
