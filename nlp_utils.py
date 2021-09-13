import re
from nltk import word_tokenize


def preprocess(text):
    text = re.sub("U.S.", "USA", text)
    text = re.sub("[^a-z ]", " ", text.lower())
    text = re.sub("photos?|videos?", "", text)
    words = word_tokenize(text)
    return words


# the vectorizer is expected to be fitted to data
def vectorize(df_column, vectorizer):
    tokens = df_column.apply(preprocess)
    return vectorizer.transform(tokens)
