from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

category = ['alt.atheism']
newsgroup_train = fetch_20newsgroups(categories=category)
count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(newsgroup_train.data)
print(count_vect.vocabulary_.get("man"))
