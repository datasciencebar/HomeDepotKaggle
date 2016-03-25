from __future__ import print_function

import gc
import numpy as np
import operator
import pandas as pd
import cPickle as pickle

from math import sqrt
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from numpy.core.umath_tests import inner1d
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

from features import get_term_match_features

# Paths to the data


BASE_PATH = "/home/dsavenk/Projects/Kaggle/HomeDepot/"
TRAIN_PATH = BASE_PATH + "train.csv"
TEST_PATH = BASE_PATH + "test.csv"
DESCRIPTIONS_PATH = BASE_PATH + "product_descriptions.csv"
ATTRIBUTES_PATH = BASE_PATH + "attributes.csv"
OUTPUT_PATH = BASE_PATH + "submission.csv"
MODEL_PATH = BASE_PATH + "gbr_model.pickle"
IDF_PATH = BASE_PATH + "idf.pickle"

STEMMER = PorterStemmer()
TSVD_DIM = 10
_tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', min_df=2)
_tsvd = TruncatedSVD(n_components=TSVD_DIM, random_state=42)
TFIDF_PIPELINE = Pipeline([('tfidf', _tfidf), ('tsvd', _tsvd), ])


def read_data(train_path, test_path, attributes_path, descriptions_path):
    """
    Reads train and test data, merges with attributes and descriptions and returns train and test data frames.
    :param train_path: Path to the train data.
    :param test_path: Path to the test data.
    :param attributes_path: Path to the attributes data.
    :param descriptions_path: Path to the descriptions data.
    :return: Train and test pandas data frames.
    """

    def aggregate_attributes(val):
        return ". ".join([row['name'] + "\t" + row['value']
                          for _, row in val[['name', 'value']].iterrows()])

    train_data = pd.read_csv(train_path, encoding="ISO-8859-1")
    test_data = pd.read_csv(test_path, encoding="ISO-8859-1")
    attributes_data = pd.read_csv(attributes_path, encoding="ISO-8859-1").dropna()
    brand_data = attributes_data[attributes_data['name'] == 'MFG Brand Name'][["product_uid", "value"]].rename(
        columns={"value": "brand"})
    attributes_data['attributes'] = " . " + attributes_data['name'] + " . " + attributes_data['value'] + " \n "
    attributes_data.drop(['name', 'value'], axis=1, inplace=True)
    attributes_data = attributes_data.groupby('product_uid', as_index=False).aggregate(np.sum)
    descriptions_data = pd.read_csv(descriptions_path, encoding="ISO-8859-1")

    all_data = pd.concat((train_data, test_data), axis=0, ignore_index=True)
    all_data = pd.merge(all_data, descriptions_data, how='left', on='product_uid')
    all_data = pd.merge(all_data, brand_data, how='left', on='product_uid')
    all_data = pd.merge(all_data, attributes_data, how='left', on='product_uid')
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
            s = s.lower()
            s = s.replace("toliet","toilet")
            s = s.replace("airconditioner","air conditioner")
            s = s.replace("vinal","vinyl")
            s = s.replace("vynal","vinyl")
            s = s.replace("skill","skil")
            s = s.replace("snowbl","snow bl")
            s = s.replace("plexigla","plexi gla")
            s = s.replace("rustoleum","rust-oleum")
            s = s.replace("whirpool","whirlpool")
            s = s.replace("whirlpoolga", "whirlpool ga")
            s = s.replace("whirlpoolstainless","whirlpool stainless")
            return [STEMMER.stem(word.lower()) for word in word_tokenize(s)]
        else:
            return []

    df = pd.concat((train_df, test_df), axis=0, ignore_index=True)
    df['title_terms'] = df['product_title'].apply(str_stem)
    df['query_terms'] = df['search_term'].apply(str_stem)
    df['description_terms'] = df['product_description'].apply(str_stem)
    df['brand_terms'] = df['brand'].apply(str_stem)
    df['attributes_terms'] = df['attributes'].apply(str_stem)
    df['product_terms'] = df['title_terms'] + df['description_terms']

    df['query_length'] = df['query_terms'].str.len()
    df['title_length'] = df['title_terms'].str.len()
    df['brand_length'] = df['brand_terms'].str.len()
    df['description_length'] = df['description_terms'].str.len()
    df = get_term_match_features(df, "query_terms", "product_terms", "product_contains")
    df = get_term_match_features(df, "query_terms", "title_terms", "title_contains")
    df['title_contains_fuzzy'] = df[['product_title', 'query_terms']].apply(
        lambda df_row: 1.0 * len([term for term in df_row['query_terms'] if term in df_row['product_title']])
        if set(df_row['query_terms']) > 0 else 0, axis=1)

    df = get_term_match_features(df, "query_terms", "description_terms", "description_contains")
    df['description_contains_fuzzy'] = df[['product_description', 'query_terms']].apply(
        lambda df_row: 1.0 * len([term for term in df_row['query_terms'] if term in df_row['product_description']])
        if set(df_row['query_terms']) > 0 else 0.0, axis=1)

    df = get_term_match_features(df, "query_terms", "brand_terms", "brand_contains")
    df = get_term_match_features(df, "query_terms", "attributes_terms", "attributes_contains")
    return df[:len(train_df)], df[len(train_df):]


def get_features_data(train_df, test_df, id_column='id',
                      label_column='relevance', exclude_columns=['id', 'product_uid', 'relevance']):
    test_ids = test_df[id_column].values
    train_labels = train_df[label_column].values

    filtered_train_df = train_df.drop(exclude_columns, axis=1)
    filtered_train_df = filtered_train_df.select_dtypes(include=['number'])
    filtered_test_df = test_df.drop(exclude_columns, axis=1)
    filtered_test_df = filtered_test_df.select_dtypes(include=['number'])

    train_document_titles = train_df['title_terms'].apply(lambda val: ' '.join(val)).values
    test_document_titles = test_df['title_terms'].apply(lambda val: ' '.join(val)).values
    train_doc_titles_features = TFIDF_PIPELINE.fit_transform(train_document_titles, train_labels)
    test_doc_titles_features = TFIDF_PIPELINE.transform(test_document_titles)
    title_tfidf_feature_names = ["title_tfidf_" + str(i) for i in xrange(TSVD_DIM)]

    train_queries = train_df['query_terms'].apply(lambda val: ' '.join(val)).values
    test_queries = test_df['query_terms'].apply(lambda val: ' '.join(val)).values
    train_queries_features = TFIDF_PIPELINE.fit_transform(train_queries, train_labels)
    test_queries_features = TFIDF_PIPELINE.transform(test_queries)
    query_tfidf_feature_names = ["query_tfidf_" + str(i) for i in xrange(TSVD_DIM)]

    train_description = train_df['description_terms'].apply(lambda val: ' '.join(val)).values
    test_description = test_df['description_terms'].apply(lambda val: ' '.join(val)).values
    train_description_features = TFIDF_PIPELINE.fit_transform(train_description, train_labels)
    test_description_features = TFIDF_PIPELINE.transform(test_description)
    description_tfidf_feature_names = ["description_tfidf_" + str(i) for i in xrange(TSVD_DIM)]

    train_attributes = train_df['attributes_terms'].apply(lambda val: ' '.join(val)).values
    test_attributes = test_df['attributes_terms'].apply(lambda val: ' '.join(val)).values
    train_attributes_features = TFIDF_PIPELINE.fit_transform(train_attributes, train_labels)
    test_attributes_features = TFIDF_PIPELINE.transform(test_attributes)
    attributes_tfidf_feature_names = ["attributes_tfidf_" + str(i) for i in xrange(TSVD_DIM)]

    query_title_cosine_train = np.einsum('ij, ij->i', train_queries_features, train_doc_titles_features)\
                               / np.linalg.norm(train_queries_features, axis=1)\
                               / np.linalg.norm(train_doc_titles_features, axis=1)
    query_title_cosine_test = np.einsum('ij, ij->i', test_queries_features, test_doc_titles_features)\
                               / np.linalg.norm(test_queries_features, axis=1)\
                               / np.linalg.norm(test_doc_titles_features, axis=1)
    query_description_cosine_train = np.einsum('ij, ij->i', train_queries_features, train_description_features)\
                               / np.linalg.norm(train_queries_features, axis=1)\
                               / np.linalg.norm(train_description_features, axis=1)
    query_description_cosine_test = np.einsum('ij, ij->i', test_queries_features, test_description_features)\
                               / np.linalg.norm(test_queries_features, axis=1)\
                               / np.linalg.norm(test_description_features, axis=1)
    # Reshape 1d arrays to row*1 matrix
    query_title_cosine_train = np.reshape(np.nan_to_num(query_title_cosine_train), (query_title_cosine_train.shape[0], 1))
    query_title_cosine_test = np.reshape(np.nan_to_num(query_title_cosine_test), (query_title_cosine_test.shape[0], 1))
    query_description_cosine_train = np.reshape(np.nan_to_num(query_description_cosine_train),
                                                (query_description_cosine_train.shape[0], 1))
    query_description_cosine_test = np.reshape(np.nan_to_num(query_description_cosine_test),
                                               (query_description_cosine_test.shape[0], 1))


    train_feature_names = np.concatenate((filtered_train_df.columns.values, title_tfidf_feature_names,
                                          query_tfidf_feature_names, description_tfidf_feature_names,
                                          attributes_tfidf_feature_names,
                                          ["query_title_tsvd_cosine", "query_description_tsvd_cosine"]))
    test_feature_names = np.concatenate((filtered_test_df.columns.values, title_tfidf_feature_names,
                                         query_tfidf_feature_names,
                                         description_tfidf_feature_names,
                                         attributes_tfidf_feature_names,
                                         ["query_title_tsvd_cosine", "query_description_tsvd_cosine"]))
    assert np.array_equal(train_feature_names, test_feature_names)

    train_features = np.concatenate((filtered_train_df.values, train_doc_titles_features, train_queries_features, 
                                     train_description_features, train_attributes_features, query_title_cosine_train,
                                     query_description_cosine_train), axis=1)
    test_features = np.concatenate((filtered_test_df.values, test_doc_titles_features, test_queries_features,
                                    test_description_features, test_attributes_features, query_title_cosine_test,
                                    query_description_cosine_test), axis=1)
    return train_labels, train_features, train_feature_names, test_features, test_ids


def train_model(labels, features, verbose=True):
    regressor = GradientBoostingRegressor()
    grid_search = GridSearchCV(estimator=regressor, param_grid={'n_estimators': (500, ),
                                                                'max_depth': (5, ),
                                                                'subsample': (0.8, ),
                                                                'learning_rate': (0.1, ), },
                               scoring='mean_squared_error', n_jobs=-1, cv=5, verbose=5)
    model = grid_search
    model.fit(features, labels)
    if verbose:
        if hasattr(model, 'best_params_'):
            print("Best parameters found by grid search:")
            print(model.best_params_)
            print("Best CV RMSE = ", sqrt(-model.best_score_))
            regressor = model.best_estimator_
        train_score = regressor.score(features, labels)
        print("Training RMSE = ", train_score)

    return regressor


def predict(model, test_ids, test_features, output_path):
    y_pred = model.predict(test_features)
    func = np.vectorize(lambda val: 3.0 if val > 3 else (1.0 if val < 1 else val))
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
    # compute_idf(pd.concat((train_df, test_df), axis=0, ignore_index=True))
    train_labels, train_features, train_feature_names, test_features, test_ids = get_features_data(train_df, test_df)
    train_df = None
    test_df = None
    gc.collect()

    model = train_model(train_labels, train_features)
    predict(model, test_ids, test_features, OUTPUT_PATH)
    with open(MODEL_PATH, 'w') as out:
        pickle.dump(model, out)
    analyze_model(model, train_feature_names)


def compute_idf(df):
    idf = dict()
    for index, row in df.iterrows():
        for term in set(row['product_terms']):
            if term not in idf:
                idf[term] = 0
            idf[term] += 1
    with open(IDF_PATH, 'w') as out:
        pickle.dump(idf, out)

if __name__ == "__main__":
    prepare_data()

    # train_df, test_df = read_data(TRAIN_PATH, TEST_PATH, ATTRIBUTES_PATH, DESCRIPTIONS_PATH)
    # train_df, test_df = generate_features(train_df, test_df)
    # df = pd.concat((train_df, test_df), axis=0, ignore_index=True)
    # compute_idf()
