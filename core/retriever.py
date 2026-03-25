from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from config import TOP_K

class Retriever:
    def __init__(self, dataset):
        self.docs = []
        for item in dataset:
            for title, content in item["context"]:
                t = " ".join(title) if isinstance(title,list) else str(title)
                c = " ".join(content) if isinstance(content,list) else str(content)
                self.docs.append(f"{t}: {c}")

        self.vec = TfidfVectorizer(stop_words="english")
        self.doc_vec = self.vec.fit_transform(self.docs)

    def search(self, query):
        q_vec = self.vec.transform([query])
        scores = (self.doc_vec @ q_vec.T).toarray().flatten()
        idx = np.argsort(scores)[-TOP_K:][::-1]
        return [self.docs[i] for i in idx]