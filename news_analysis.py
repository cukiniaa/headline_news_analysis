import _pickle as cPickle
import pandas as pd
import matplotlib.pyplot as plt
from time import time

pickle_filename = r'classification.sav'

with open(pickle_filename, "rb") as f:
    clf = cPickle.load(f)
    vectorizer = cPickle.load(f)
    cat_dict = cPickle.load(f)

filename = 'data/abcnews-date-text.csv'
df = pd.read_csv(filename)
print(df.info())
print(df.head())

print("Tokenizing headlines", end=" ")
t0 = time()
tokens = vectorizer.preprocess_tokenize(df['headline_text'])
print("%.3f (s)" % time()-t0)

print("Vectorizing tokens", end=" ")
t0 = time()
X = vectorizer.transform(tokens)
print("%.3f (s)" % time()-t0)

print("Predicting categories", end=" ")
t0 = time()
my_pred = clf.predict(X)
print("%.3f (s)" % time()-t0)

# change codes to labels and plot
df['category'] = [cat_dict[p] for p in my_pred]
df['category'].value_counts().plot.barh(figsize=(10, 10))
plt.draw()

label = 'CULTURE'
max_ind = 20
print('Printing first %d headlines from category %s.' % (max_ind, label))
for headline in df[df['category'] == label]['headline_text'][:20]:
    print(headline)


out_filename = 'data/classified-abcnews-date-text.csv'
df.to_csv(out_filename, index=False)

plt.show()
