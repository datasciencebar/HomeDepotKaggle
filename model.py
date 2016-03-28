from __future__ import print_function

import gc
import numpy as np
import operator
import pandas as pd
import dill
import cPickle as pickle

from math import sqrt
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline, FeatureUnion

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


class KeepNumericFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Removes non-numeric columns from pandas data frame and returns the corresponding numpy array.
    """
    def __init__(self, exclude_columns=None):
        if exclude_columns is None:
            exclude_columns = ['id', 'product_uid', 'relevance']
        self.exclude_columns = exclude_columns

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df.drop(self.exclude_columns, axis=1).select_dtypes(include=['number']).values


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Keeps only the given list of columns in a data frame.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df[self.columns]


class ColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Applies the given function to the given list of columns in a data frame.
    """
    def __init__(self, columns, func):
        self.columns = columns
        self.func = func

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df[self.columns].apply(self.func)


class CosineSimilarityTransformer(BaseEstimator, TransformerMixin):
    """
    Compute cosine similarity between vectors stored inside the features matrix.
    """
    def __init__(self, query_vector_range, doc_vectors_ranges):
        self.query_vector_range = query_vector_range
        self.doc_vectors_ranges = doc_vectors_ranges

    def fit(self, x, y=None):
        return self

    def transform(self, features):
        query_vector = features[:,self.query_vector_range[0]:self.query_vector_range[1]]
        doc_vectors = []
        for rng in self.doc_vectors_ranges:
            doc_vectors.append(features[:, rng[0]:rng[1]])
        res = [features, ]
        for doc_vector in doc_vectors:
            res.append(self._compute_cosine(query_vector, doc_vector))
        return np.hstack(res)

    def _compute_cosine(self, vec_1, vec_2):
        res = np.einsum('ij, ij->i', vec_1, vec_2) / np.linalg.norm(vec_1, axis=1) / np.linalg.norm(vec_2, axis=1)
        return np.reshape(np.nan_to_num(res), (res.shape[0], 1))



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
    df = get_term_match_features(df, "query_terms", "description_terms", "description_contains")
    df = get_term_match_features(df, "query_terms", "brand_terms", "brand_contains")
    df = get_term_match_features(df, "query_terms", "attributes_terms", "attributes_contains")

    # Drop raw text columns.
    df.drop(['product_title', 'search_term', 'product_description', 'attributes'], axis=1, inplace=True)

    return df[:len(train_df)], df[len(train_df):]


def train_model(labels, features, verbose=True):
    regressor = GradientBoostingRegressor(n_estimators=500, max_depth=5, subsample=0.8, learning_rate=0.1)
    identity = lambda x: x

    tsvd_dimension = 10

    model_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('topic', Pipeline([
                ('comp_topics', FeatureUnion([
                    ('qtopic', Pipeline([
                        ('get_terms', ColumnSelector('query_terms')),
                        ('tfidf', TfidfVectorizer(preprocessor=identity, tokenizer=identity, stop_words='english', min_df=1)),
                        ('topic', TruncatedSVD(n_components=tsvd_dimension))
                    ])),
                    ('ttopic', Pipeline([
                        ('get_terms', ColumnSelector('title_terms')),
                        ('tfidf', TfidfVectorizer(preprocessor=identity, tokenizer=identity, stop_words='english', min_df=1)),
                        ('topic', TruncatedSVD(n_components=tsvd_dimension))
                    ])),
                    ('dtopic', Pipeline([
                        ('get_terms', ColumnSelector('description_terms')),
                        ('tfidf', TfidfVectorizer(preprocessor=identity, tokenizer=identity, stop_words='english', min_df=1)),
                        ('topic', TruncatedSVD(n_components=tsvd_dimension))
                    ])),
                    ('atopic', Pipeline([
                        ('get_terms', ColumnSelector('description_terms')),
                        ('tfidf', TfidfVectorizer(preprocessor=identity, tokenizer=identity, stop_words='english', min_df=1)),
                        ('topic', TruncatedSVD(n_components=tsvd_dimension))
                    ]))
                ])),
                ('cosine', CosineSimilarityTransformer((0, tsvd_dimension), [(beg, beg + tsvd_dimension)
                                                                             for beg in xrange(tsvd_dimension,
                                                                                               3 * tsvd_dimension,
                                                                                               tsvd_dimension)]))
            ])),
            ('match', KeepNumericFeaturesTransformer())
        ])),
        ('regr', regressor)
    ])

    grid_search = GridSearchCV(estimator=model_pipeline, param_grid={'regr__n_estimators': (500, ),
                                                                'regr__max_depth': (5, ),
                                                                'regr__subsample': (0.8, ),
                                                                'regr__learning_rate': (0.1, ), },
                               scoring='mean_squared_error', n_jobs=1, cv=5, verbose=5)

    grid_search.fit(features, labels)
    if verbose:
        if hasattr(grid_search, 'best_params_'):
            print("Best parameters found by grid search:")
            print(grid_search.best_params_)
            print("Best CV RMSE = ", sqrt(-grid_search.best_score_))
        train_score = grid_search.best_estimator_.score(features, labels)
        print("Training RMSE = ", train_score)

    return grid_search


def predict(model, test_ids, test_features, output_path):
    y_pred = model.predict(test_features)
    func = np.vectorize(lambda val: 3.0 if val > 3 else (1.0 if val < 1 else val))
    y_pred = func(y_pred)
    pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv(output_path, index=False)


def analyze_model(model, features=None):
    model = model.best_estimator_.named_steps['regr']
    if hasattr(model, 'feature_importances_'):
        if features is None:
            features = [str(i) for i in xrange(len(model.feature_importances_))]
        feature_scores = zip(features, model.feature_importances_)
        feature_scores.sort(key=operator.itemgetter(1), reverse=True)
        print("Feature importances:")
        print("\n".join([name + ": " + str(score) for name, score in feature_scores]))


def prepare_data():
    train_df, test_df = read_data(TRAIN_PATH, TEST_PATH, ATTRIBUTES_PATH, DESCRIPTIONS_PATH)
    train_df, test_df = generate_features(train_df, test_df)
    # compute_idf(pd.concat((train_df, test_df), axis=0, ignore_index=True))

    test_ids = test_df['id'].values
    train_labels = train_df['relevance'].values

    gc.collect()
    model = train_model(train_labels, train_df)
    predict(model, test_ids, test_df, OUTPUT_PATH)
    with open(MODEL_PATH, 'w') as out:
        dill.dump(model, out)
    analyze_model(model)


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
