import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class HeadlinesVectorizer(CountVectorizer):

    def __init__(self, **vec_args):
        params = self._default_params | vec_args
        # HashingVectorizer.__init__(self, **params)
        CountVectorizer.__init__(self, **params)
        self._transformer = TfidfTransformer()

    def preprocess_tokenize(self, data):

        stopwords_set = set(stopwords.words('english'))

        def text_to_tokens(text):
            text = re.sub("U.S.", "USA", text)
            text = re.sub("[^a-z ]", " ", text.lower())
            text = re.sub("photos?|videos?", "", text)
            words = word_tokenize(text)
            words = [w for w in words if len(w) > 2 and w not in stopwords_set]
            return words

        return data.apply(text_to_tokens)

    def fit_transform(self, raw_documents, y=None):
        X = CountVectorizer.fit_transform(self, raw_documents, y)
        self._transformer.fit(X)
        return self._transformer.transform(X)

    def transform(self, raw_documents):
        X = CountVectorizer.transform(self, raw_documents)
        return self._transformer.transform(X)

    def _identity(v):
        return v

    _default_params = dict(
        preprocessor=_identity,
        tokenizer=_identity,
        lowercase=False,
        token_pattern=None,
        max_features=50000,
        ngram_range=(1, 2)
    )
