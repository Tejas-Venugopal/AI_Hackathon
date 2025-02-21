# utils/vectorizer.py
from sentence_transformers import SentenceTransformer
import numpy as np

class Vectorizer:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
    def vectorize(self, text):
        return self.model.encode(text, convert_to_numpy=True)

# utils/retriever.py
import faiss
import numpy as np

class Retriever:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.index = faiss.IndexHNSWFlat(768, 32)
        self.documents = []
        
    def add_documents(self, documents):
        embeddings = self.vectorizer.vectorize(documents)
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(documents)
        
    def retrieve(self, query, k=3):
        query_embedding = self.vectorizer.vectorize(query)
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        return [self.documents[i] for i in indices[0]]

# models/faq_model.py
