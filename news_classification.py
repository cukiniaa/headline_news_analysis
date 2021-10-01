import json
import pandas as pd

import nltk
from nlp_utils import HeadlinesVectorizer

from collections import Counter
from pandas.core.common import flatten
from sklearn.utils import resample

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn import metrics
from time import time

from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt

import _pickle as cPickle

nltk.download('stopwords')
nltk.download('punkt')


def test_classifier(clf):
    train_t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - train_t0

    pred_t0 = time()
    pred = clf.predict(X_test)
    pred_time = time() - pred_t0

    acc = metrics.accuracy_score(y_test, pred)
    prec = metrics.precision_score(y_test, pred, average='weighted')
    rec = metrics.recall_score(y_test, pred, average='weighted')
    f1 = metrics.f1_score(y_test, pred, average='weighted')

    print(clf.__class__.__name__)
    print("SCORES Accuracy: %.3f. Precision: %.3f. Recall: %.3f. F1: %.3f."
          % (acc, prec, rec, f1))
    print("TRAINING TIME %.3f s." % train_time)
    print("PREDICTION TIME %.3f s." % pred_time)
    print("-------------------------------")

    return pred


def binary_class_precision(y_test, pred, codes_dict, print_scores=False):
    precision = {}
    for code, label in codes_dict.items():
        bin_y_test = y_test == code
        bin_pred = pred == code
        precision[label] = metrics.precision_score(bin_y_test, bin_pred)
        if print_scores:
            print("%s %.5f" % (label, precision[label]))
    return precision


filename = 'data/News_Category_Dataset_v2.json'
plot_truncatedSVD = False

# ----------------------------------------
print("............................")
print("\n>>> Reading in data")

rows = []

with open(filename) as f:
    for line in f:
        rows.append(json.loads(line))

data = pd.DataFrame(rows)
data = data.drop(columns=['authors', 'link', 'short_description', 'date'])
data['category'] = data['category'].astype('category')

plt.figure()
data['category'].value_counts().plot.barh(figsize=(10, 10))
plt.title("Categories in the dataset")
plt.tight_layout()
plt.draw()

print(data.info())

# ----------------------------------------
print("\n>>> Merging similar categories")

new_categories = {
    'ENVIRONMENT': ['GREEN', 'ENVIRONMENT'],
    'GROUPS VOICES': ['LATINO VOICES', 'BLACK VOICES', 'QUEER VOICES'],
    'CULTURE': ['CULTURE & ARTS', 'ARTS & CULTURE', 'ARTS', 'RELIGION'],
    'FINANCE': ['BUSINESS', 'MONEY'],
    'SCIENCE & TECH': ['SCIENCE', 'TECH'],
    'EDUCATION': ['EDUCATION', 'COLLEGE'],
    'WORLD NEWS': ['WORLD NEWS', 'THE WORLDPOST', 'WORLDPOST'],
    'ENTERTAINMENT': ['ENTERTAINMENT', 'COMEDY'],
    'WELLNESS': ['HEALTHY LIVING', 'WELLNESS'],
    'FOOD & DRINK': ['FOOD & DRINK', 'TASTE'],
    'STYLE & BEAUTY': ['STYLE & BEAUTY', 'STYLE'],
    'RELATIONSHIPS': ['PARENTING', 'PARENTS', 'WEDDINGS', 'DIVORCE'],
    'OTHER': ['GOOD NEWS', 'WEIRD NEWS', 'FIFTY']
}

mapper = {}
for key, values in new_categories.items():
    for val in values:
        mapper[val] = key

data = data.replace({'category': mapper})
data['category'] = data['category'].astype('category')

plt.figure()
data['category'].value_counts().plot.barh(figsize=(10, 10))
plt.title("Categories after merging")
plt.tight_layout()
plt.draw()

print("NUmber of categories %d." % len(data['category'].unique()))
print("Number of all headlines %d." % len(data))

# ----------------------------------------
print("\n>>> Preprocessing and tokenizing")

vectorizer = HeadlinesVectorizer()

data['headline'] = vectorizer.preprocess_tokenize(data['headline'])

grouped = data.groupby(by='category', as_index=False).agg(
    lambda a: list(flatten(a)))
grouped['headline'] = [Counter(h_a) for h_a in grouped['headline']]
grouped['common_words'] = [sorted(group.items(), key=lambda x: x[1],
                                  reverse=True)[:10]
                           for group in grouped['headline']]


print("Most common words per category:")
for _, row in grouped.iterrows():
    print(row['category'], row['common_words'])

# ----------------------------------------
print("\n>>> Handling class umbalance.")

value_counts = data['category'].value_counts()
sorted_keys = value_counts.keys()
rows_per_cat = value_counts[sorted_keys[0]]

data_upsampled = pd.DataFrame(data[data['category'] == sorted_keys[0]])

for key in sorted_keys[1:]:
    resampled = resample(data[data['category'] == key],
                         replace=True,
                         n_samples=rows_per_cat)
    data_upsampled = pd.concat([data_upsampled, resampled])

data = data_upsampled

# ----------------------------------------
print("\n>>> Vectorizing the corpus")

X = vectorizer.fit_transform(data['headline'])
y = data['category'].cat.codes

# cat_dict { code: label }
cat_dict = dict(enumerate(data['category'].cat.categories))

if plot_truncatedSVD:
    pca = TruncatedSVD(n_components=2)
    X_pca = pca.fit(X)
    X_pca_scores = X_pca.transform(X)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=X_pca_scores[:, 0], y=X_pca_scores[:, 1],
                    hue=data['category'])
    plt.tight_layout()
    plt.draw()

print("Shape of X:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# how to print the vocabulary if the vectorizer doesn't hash
# d = dict(zip(vectorizer.get_feature_names(),
#              np.array(X.sum(axis=0)).flatten()))
# sorted_d = sorted(d.items(), key=lambda kv: kv[1], reverse=False)

# ----------------------------------------
print("\n>>> Fitting the data")

clf = LinearSVC()
pred = test_classifier(clf)

codes_dict = dict(enumerate(data['category'].cat.categories))
binary_class_precision(y_test, pred, codes_dict, print_scores=True)
print("-------------------------------")

# ----------------------------------------
print("\n>>> Saving the model and the vectorizer")

pickle_filename = r'classification.sav'

with open(pickle_filename, 'wb') as f:
    cPickle.dump(clf, f)
    cPickle.dump(vectorizer, f)
    cPickle.dump(cat_dict, f)
    print("Model, vectorizer and dict[code] = label saved in", pickle_filename)

plt.show()
