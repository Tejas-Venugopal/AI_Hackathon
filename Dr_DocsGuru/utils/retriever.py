import faiss
import numpy as np

class Retriever:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.index = faiss.IndexFlatL2(768)
        self.docs = []

    def add_documents(self, docs):
        vectors = [self.vectorizer.vectorize(doc) for doc in docs]
        self.index.add(np.vstack(vectors))
        self.docs.extend(docs)

    def retrieve(self, query, k=1):
        query_vector = self.vectorizer.vectorize(query)
        _, indices = self.index.search(query_vector, k)
        return [self.docs[idx] for idx in indices[0]]