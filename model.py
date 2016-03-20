from __future__ import print_function

import nltk
import numpy as np
import operator
import pandas as pd
import cPickle as pickle

from math import sqrt
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.grid_search import GridSearchCV

# Paths to the data
from features import get_term_match_features

BASE_PATH = "/home/dsavenk/Projects/Kaggle/HomeDepot/"
TRAIN_PATH = BASE_PATH + "train.csv"
TEST_PATH = BASE_PATH + "test.csv"
DESCRIPTIONS_PATH = BASE_PATH + "product_descriptions.csv"
ATTRIBUTES_PATH = BASE_PATH + "attributes.csv"
OUTPUT_PATH = BASE_PATH + "submission.csv"
MODEL_PATH = BASE_PATH + "gbr_model.pickle"

STEMMER = PorterStemmer()


def read_data(train_path, test_path, attributes_path, descriptions_path):
    """
    Reads train and test data, merges with attributes and descriptions and returns train and test data frames.
    :param train_path: Path to the train data.
    :param test_path: Path to the test data.
    :param attributes_path: Path to the attributes data.
    :param descriptions_path: Path to the descriptions data.
    :return: Train and test pandas data frames.
    """
    train_data = pd.read_csv(train_path, encoding="ISO-8859-1")
    test_data = pd.read_csv(test_path, encoding="ISO-8859-1")
    attributes_data = pd.read_csv(attributes_path, encoding="ISO-8859-1")
    descriptions_data = pd.read_csv(descriptions_path, encoding="ISO-8859-1")
    brand_data = attributes_data[attributes_data['name'] == 'MFG Brand Name'][["product_uid", "value"]].rename(
        columns={"value": "brand"})
    all_data = pd.concat((train_data, test_data), axis=0, ignore_index=True)
    all_data = pd.merge(all_data, descriptions_data, how='left', on='product_uid')
    all_data = pd.merge(all_data, brand_data, how='left', on='product_uid')
    return all_data.iloc[:len(train_data)], all_data.iloc[len(train_data):]


def fix_units(s, replacements = {"'|in|inches|inch": "in",
                    "''|ft|foot|feet": "ft",
                    "pound|pounds|lb|lbs": "lb",
                    "volt|volts|v": "v",
                    "watt|watts|w": "w",
                    "ounce|ounces|oz": "oz",
                    "gal|gallon": "gal",
                    "m|meter|meters": "m",
                    "cm|centimeter|centimeters": "cm",
                    "mm|milimeter|milimeters": "mm",
                    "yd|yard|yards": "yd",
                    }):

    regexp_template = r"([/\.0-9]+)[-\s]*({0})([,\.\s]|$)"
    regexp_subst_template = "\g<1> {0} "

    s = re.sub(r"([^\s-])x([0-9]+)", "\g<1> x \g<2>", s).strip()

    for pattern, repl in replacements.iteritems():
        s = re.sub(regexp_template.format(pattern), regexp_subst_template.format(repl), s)

    s = re.sub(r"\s\s+", " ", s).strip()
    return s


def generate_features(train_df, test_df):
    def str_stem(s):
        if isinstance(s, str) or isinstance(s, unicode):
            s = re.sub('[^A-Za-z0-9-./]', ' ', s.lower())
            s = fix_units(s)
            return [STEMMER.stem(word.lower()) for word in word_tokenize(s)]
        else:
            return []

    df = pd.concat((train_df, test_df), axis=0, ignore_index=True)
    df['title_terms'] = df['product_title'].apply(str_stem)
    df['query_terms'] = df['search_term'].apply(str_stem)
    df['description_terms'] = df['product_description'].apply(str_stem)
    df['brand_terms'] = df['brand'].apply(str_stem)
    df['product_terms'] = df['title_terms'] + df['description_terms']

    df['query_length'] = df['query_terms'].str.len()
    df['title_length'] = df['title_terms'].str.len()
    df['brand_length'] = df['brand_terms'].str.len()
    df['description_length'] = df['description_terms'].str.len()
    get_term_match_features(df, "query_terms", "product_terms", "product_contains")
    get_term_match_features(df, "query_terms", "title_terms", "title_contains")
    df['title_contains_fuzzy'] = df[['product_title', 'query_terms']].apply(
        lambda df_row: 1.0 * len([term for term in df_row['query_terms'] if term in df_row['product_title']])
        if set(df_row['query_terms']) > 0 else 0, axis=1)

    get_term_match_features(df, "query_terms", "description_terms", "description_contains")
    df['description_contains_fuzzy'] = df[['product_description', 'query_terms']].apply(
        lambda df_row: 1.0 * len([term for term in df_row['query_terms'] if term in df_row['product_description']])
        if set(df_row['query_terms']) > 0 else 0.0, axis=1)

    get_term_match_features(df, "query_terms", "brand_terms", "brand_contains")
    return df[:len(train_df)], df[len(train_df):]


def get_features_data(df, id_column='id', label_column='relevance', exclude_columns=['id', 'product_uid', 'relevance']):
    ids = df[id_column].values
    labels = df[label_column].values
    filtered_df = df.drop(exclude_columns, axis=1)
    filtered_df = filtered_df.select_dtypes(include=['number'])
    feature_names = filtered_df.columns.values
    features = filtered_df.values
    return ids, labels, features, feature_names


def train_model(labels, features, verbose=True):
    regressor = GradientBoostingRegressor()
    grid_search = GridSearchCV(estimator=regressor, param_grid={'n_estimators': (100, 250, 500, 1000),
                                                                'max_depth': (3, 5, 6),
                                                                'learning_rate': (0.01, 0.1),},
                               scoring='mean_squared_error', n_jobs=-1, cv=5)
    model = grid_search
    model.fit(features, labels)
    if verbose:
        if hasattr(model, 'best_params_'):
            print("Best parameters found by grid search:")
            print(model.best_params_)
            print("Best CV RMSE = ", sqrt(-model.best_score_))
            regressor = model.best_estimator_
        train_score = regressor.score(features, labels)
        print("Training RMSE = ", sqrt(train_score))

    return regressor


def predict(model, test_ids, test_features, output_path):
    y_pred = model.predict(test_features)
    func = np.vectorize(lambda val: 3.0 if val > 3 else val)
    y_pred = func(y_pred)
    pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv(output_path, index=False)


def analyze_model(model, features):
    if hasattr(model, 'feature_importances_'):
        feature_scores = zip(features, model.feature_importances_)
        feature_scores.sort(key=operator.itemgetter(1), reverse=True)
        print("Feature importances:")
        print("\n".join([name + ": " + str(score) for name, score in feature_scores]))


def prepare_data():
    train_df, test_df = read_data(TRAIN_PATH, TEST_PATH, ATTRIBUTES_PATH, DESCRIPTIONS_PATH)
    train_df, test_df = generate_features(train_df, test_df)
    _, labels, train_features, train_feature_names = get_features_data(train_df)
    test_ids, _, test_features, test_feature_names = get_features_data(test_df)
    assert np.array_equal(train_feature_names, test_feature_names)
    model = train_model(labels, train_features)
    predict(model, test_ids, test_features, OUTPUT_PATH)
    with open(MODEL_PATH, 'w') as out:
        pickle.dump(model, out)
    analyze_model(model, train_feature_names)


def compute_idf():
    train_df, test_df = read_data(TRAIN_PATH, TEST_PATH, ATTRIBUTES_PATH, DESCRIPTIONS_PATH)
    train_df, test_df = generate_features(train_df, test_df)
    df = pd.concat((train_df, test_df), axis=0, ignore_index=True)

    idf = dict()
    for index, row in df.iterrows():
        for term in set(row['product_terms']):
            if term not in idf:
                idf[term] = 0
            idf[term] += 1
    with open(BASE_PATH + 'idf.pickle', 'w') as out:
        pickle.dump(idf, out)

if __name__ == "__main__":
    prepare_data()
    # compute_idf()
