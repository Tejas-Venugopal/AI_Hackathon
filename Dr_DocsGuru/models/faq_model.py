from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

class FAQModel:
    def __init__(self, retriever):
        self.retriever = retriever
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")
        self.load_documents()

    def load_documents(self):
        docs = []
        for i in range(1, 10):
            with open(f'data/faq_docs/doc{i}.txt', 'r') as file:
                docs.append(file.read())
        self.retriever.add_documents(docs)

    def get_answer(self, question):
        inputs = self.tokenizer([question], return_tensors='pt')
        question_hidden_states = self.model.question_encoder(input_ids=inputs['input_ids'])[0]
        doc_scores = self.model.retriever(question_hidden_states, n_docs=1)
        doc_tokens = self.model.retriever.postprocess_gathered_doc_tokens(doc_scores)
        outputs = self.model.generate(context_input_ids=doc_tokens['input_ids'], context_attention_mask=doc_tokens['attention_mask'])
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]