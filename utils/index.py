import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# df = pd.read_csv("../data/sampah_bandung.csv")
# start_time = datetime.datetime.now()
def search_data(query, df):
    vectorize = TfidfVectorizer()
    X = vectorize.fit_transform(df["text"].iloc[:])
    query = vectorize.transform([query])
    results = cosine_similarity(query, X).flatten()
    return results


# end_time = datetime.datetime.now()

# results = search_data("sampah", df)
# for i in results.argsort()[-5:][::-1]:
#     print(i)


# print("Time taken: {}".format(end_time - start_time))
