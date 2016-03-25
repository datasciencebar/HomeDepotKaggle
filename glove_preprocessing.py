
import codecs
import cPickle as pickle

from nltk.stem.porter import PorterStemmer
from sys import argv


if __name__ == "__main__":
    with open(argv[3], 'r') as inp:
        idf = pickle.load(inp)
    stemmer = PorterStemmer()
    with codecs.open(argv[1], 'r', 'utf8') as inp, codecs.open(argv[2], 'w', 'utf8') as out:
        for line in inp:
            fields = line.split(' ')
            if len(fields) != 301:
                print fields
                continue
            if fields[0] in idf:
                fields[0] = stemmer.stem(fields[0].lower())
                print >> out, "\t".join(fields)
