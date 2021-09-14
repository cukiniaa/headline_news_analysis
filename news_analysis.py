import _pickle as cPickle
import pandas as pd
import matplotlib.pyplot as plt
from nlp_utils import vectorize

pickle_filename = r'classification.sav'

# TODO create a vectorizer class to hide this function (had to put it when loading 
# the vectorizer from pickle)
def identity(v):
    return v

with open(pickle_filename, "rb") as f:
    clf = cPickle.load(f)
    vectorizer = cPickle.load(f)
    cat_dict = cPickle.load(f)

filename = 'data/abcnews-date-text.csv'
df = pd.read_csv(filename)
print(df.info())
print(df.head())

X = vectorize(df['headline_text'], vectorizer)
my_pred = clf.predict(X)

# change codes to labels and plot
df['category'] = [cat_dict[p] for p in my_pred]
df['category'].value_counts().plot.barh(figsize=(10, 10))
plt.draw()

label = 'CULTURE'
max_ind = 20
print('Printing first %d headlines from category %s.' % (max_ind, label))
for headline in df[df['category'] == label]['headline_text'][:20]:
    print(headline)

plt.show()
