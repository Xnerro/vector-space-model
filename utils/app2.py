from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from numpy.linalg import norm
import re
import datetime


# Inisiasi class
class VSM:
    def __init__(self, data):
        self.data = data
        self.vectorizer = TfidfVectorizer()
        self.tf_transformer = TfidfTransformer()
        self.count_vectorizer = CountVectorizer()

    def word_count(self):
        X = self.count_vectorizer.fit_transform(self.data)
        return X.toarray()

    def tf_idf(self, word_count):
        X = self.tf_transformer.fit_transform(word_count)
        return X.toarray()

    def initial_tokens(self):
        df = pd.DataFrame({"doc": [x for x in self.data]})
        df["tokens"] = [re.split(r"\W+", x.lower()) for x in df["doc"]]
        return df

    def normalized(self, table):
        sqrt_vec = np.sqrt(table.pow(2).sum(axis=1))
        return table.div(sqrt_vec, axis=0)

    def search(self, query):
        X = self.vectorizer.fit_transform(self.data)
        query = self.vectorizer.transform([query])
        results = cosine_similarity(query, X).flatten()
        for i in results.argsort()[-5:][::-1]:
            print(df.iloc[i, 0], "--", df.iloc[i, 1], end="\n\n")
        return results

    @staticmethod
    def cosine(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))


data = pd.read_csv("sampah_bandung.csv")

korpus = [x for x in data["text"].iloc[:10]]

try:
    start_time = datetime.datetime.now()
    new_vsm = VSM(korpus)
    word_count = new_vsm.word_count()
    tf = pd.DataFrame(word_count, columns=new_vsm.count_vectorizer.get_feature_names_out())
    X = new_vsm.tf_idf(tf)
    idf = pd.DataFrame({"idf": new_vsm.tf_transformer.idf_, "term": new_vsm.count_vectorizer.get_feature_names_out()})
    tf_idf = pd.DataFrame(X, columns=new_vsm.count_vectorizer.get_feature_names_out())
    df = new_vsm.initial_tokens()
    tf = df.tokens.apply(lambda x: pd.Series(x).value_counts()).fillna(0)
    tf.sort_index(inplace=True, axis=1)
    idf = pd.Series(
        [
            np.log((float(df.shape[0]) + 1) / (len([x for x in df.tokens.values if token in x]) + 1)) + 1
            for token in tf.columns
        ]
    )
    idf.index = tf.columns
    W_td = tf.copy()
    for col in W_td.columns:
        W_td[col] = W_td[col] * idf[col]
    W_td = new_vsm.normalized(W_td)
    new_df = pd.DataFrame()
    results = new_vsm.search("minggu pagi")
    end_time = datetime.datetime.now()
    print("Waktu eksekusi: {}".format(end_time - start_time))
except Exception as e:
    print(e)
