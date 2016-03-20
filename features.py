
import pandas as pd
import cPickle as pickle

from Levenshtein import distance
from math import log, sqrt

BASE_PATH = "/home/dsavenk/Projects/Kaggle/HomeDepot/"
IDF_PATH = BASE_PATH + "idf.pickle"
N = 166694 + 74068

_idf = None


def get_idf(term):
    global _idf
    if _idf is None:
        with open(IDF_PATH, 'r') as inp:
            _idf = pickle.load(inp)
    return log(1 + ((1.0 * N / _idf[term]) if term in _idf else 0.0))


def get_term_match_features(df, query_column, content_column, result_column):
    df[result_column] = df[[content_column, query_column]].apply(
        lambda df_row: 1.0 * len(set(df_row[content_column]).intersection(set(df_row[query_column]))), axis=1)
    df[result_column + "_lev"] = df[[content_column, query_column]].apply(
        lambda df_row: sum(1 for term1 in df_row[query_column] for term2 in df_row[content_column]
                           if distance(term1, term2) < 0.3 * min(len(term1), len(term2))), axis=1)

    df[result_column + "_qnorm"] = df[[content_column, query_column]].apply(
        lambda df_row: 1.0 * len(set(df_row[content_column]).intersection(set(df_row[query_column]))) /
                       len(set(df_row[query_column]))
        if len(set(df_row[query_column])) > 0 else 0.0, axis=1)
    df[result_column + "_lev"] = df[[content_column, query_column]].apply(
        lambda df_row: 1.0 * sum(1 for term1 in df_row[query_column] for term2 in df_row[content_column]
                           if distance(term1, term2) < 0.3 * min(len(term1), len(term2))) /
                       len(set(df_row[query_column]))
        if len(set(df_row[query_column])) > 0 else 0.0, axis=1)

    # Generate tf-idf based features
    if "sorted" not in query_column:
        df[result_column + "_idf"] = df[[content_column, query_column]].apply(
            lambda df_row: sum([get_idf(term)
                                for term in set(df_row[content_column]).intersection(set(df_row[query_column]))]),
            axis=1)
        df[result_column + "_idf_cosine"] = df[[content_column, query_column]].apply(
            lambda df_row: sum([get_idf(term) * get_idf(term)
                                for term in set(df_row[content_column]).intersection(set(df_row[query_column]))]) /
                           (1 + sqrt(sum([get_idf(term) * get_idf(term) for term in set(df_row[content_column])])) *
                            (sqrt(sum([get_idf(term) * get_idf(term) for term in set(df_row[query_column])])))),
            axis=1)


if __name__ == "__main__":
    pass