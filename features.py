
import numpy as np
import pandas as pd
import cPickle as pickle

from Levenshtein import distance
from math import log, sqrt

BASE_PATH = "/home/dsavenk/Projects/Kaggle/HomeDepot/"
IDF_PATH = BASE_PATH + "idf.pickle"
EMBEDDINGS_PATH = BASE_PATH + "embeddings.txt"

N = 166694 + 74068

_idf = None
_embeddings = None


def read_embeddings():
    res = dict()
    with open(EMBEDDINGS_PATH, 'r') as inp:
        for line in inp:
            line = line.split('\t')
            term = line[0]
            vec = np.array(map(float, line[1:]))
            if term not in res:
                res[term] = []
            res[term].append(vec)
    return res

def cosine(vec_1, vec_2):
    return np.nan_to_num(np.dot(vec_1, vec_2) / np.linalg.norm(vec_1) / np.linalg.norm(vec_2))

def get_embedding(term):
    global _embeddings
    if _embeddings is None:
        _embeddings = read_embeddings()
        _embeddings = dict([(term, np.mean(np.array(vectors), axis=0)) for term, vectors in _embeddings.iteritems()])
    return _embeddings[term] if term in _embeddings else None

def get_embedding_similarity(term_a, term_b):
    emb_a = get_embedding(term_a)
    emb_b = get_embedding(term_b)
    if emb_a is not None and emb_b is not None:
        return cosine(emb_a, emb_b)
    else:
        return 1 if term_a == term_b else 0


def get_avg_embedding(terms):
    term_embeddings = filter(lambda x: x is not None, [get_embedding(term) for term in terms])
    return np.average(term_embeddings) if term_embeddings else np.array([0, ] * len(get_embedding(".")))


def get_idf(term):
    global _idf
    if _idf is None:
        with open(IDF_PATH, 'r') as inp:
            _idf = pickle.load(inp)
    return log(1 + ((1.0 * N / _idf[term]) if term in _idf else 0.0))


def get_ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])


def get_term_match_features(df, query_column, content_column, result_column):
    df[result_column] = df[[content_column, query_column]].apply(
        lambda df_row: 1.0 * len(set(df_row[content_column]).intersection(set(df_row[query_column]))), axis=1)
    df[result_column + "_ngram"] = df[[content_column, query_column]].apply(
        lambda df_row: 1.0 * len(set(get_ngrams(df_row[content_column], 2)).intersection(
            set(get_ngrams(df_row[query_column], 2)))), axis=1)
    df[result_column + "_qnorm"] = df[[content_column, query_column]].apply(
        lambda df_row: 1.0 * len(set(df_row[content_column]).intersection(set(df_row[query_column]))) /
                       len(set(df_row[query_column]))
        if len(set(df_row[query_column])) > 0 else 0.0, axis=1)
    df[result_column + "_lev"] = df[[content_column, query_column]].apply(
        lambda df_row: sum([1 if len([1 for content_term in df_row[content_column]
                             if distance(query_term, content_term) < 0.3 * min(len(query_term), len(content_term))]) > 0 else 0
                        for query_term in df_row[query_column]]), axis=1)
    df[result_column + "_embed"] = df[[content_column, query_column]].apply(
        lambda df_row: np.average([max([get_embedding_similarity(query_term, content_term)
                                        for content_term in df_row[content_column]])
                        for query_term in df_row[query_column]])
        if len(df_row[content_column]) > 0 and len(df_row[query_column]) > 0 else 0.0, axis=1)
    df[result_column + "_avg_embed"] = df[[content_column, query_column]].apply(
        lambda df_row: cosine(get_avg_embedding(df_row[query_column]),
                              get_avg_embedding(df_row[content_column])), axis=1)
    df[result_column + "_lev_norm"] = df[[content_column, query_column]].apply(
        lambda df_row: 1.0 * sum(1 for term1 in df_row[query_column] for term2 in df_row[content_column]
                           if distance(term1, term2) < 0.3 * min(len(term1), len(term2))) /
                       len(set(df_row[query_column]))
        if len(set(df_row[query_column])) > 0 else 0.0, axis=1)

    # Generate tf-idf based features
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
    return df


if __name__ == "__main__":
    pass