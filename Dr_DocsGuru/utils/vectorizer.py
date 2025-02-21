import numpy as np
from transformers import AutoTokenizer, AutoModel

class Vectorizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')
        self.model = AutoModel.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')

    def vectorize(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()