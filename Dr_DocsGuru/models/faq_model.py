from transformers import AutoTokenizer, AutoModelForCausalLM
from threading import Lock
import os

class FAQModel:
    def __init__(self, retriever):
        self.retriever = retriever
        self.model = None
        self.tokenizer = None
        self.lock = Lock()
        self.load_model()
        self.load_documents()

    def load_model(self):
        with self.lock:
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
            self.model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1",
                device_map="auto",
                load_in_4bit=True
            )
            
    def load_documents(self):
        faq_docs = []
        for company in ['Bluestar', 'CCS', 'Conifer', 'CVS', 'Digitiva', 
                       'HealthFirst', 'Lilly', 'MDLive', 'Welldoc']:
            path = f"data/faq_docs/{company}FAQ.txt"
            if os.path.exists(path):
                with open(path, 'r') as f:
                    faq_docs.append(f.read())
        self.retriever.add_documents(faq_docs)

    def format_prompt(self, question, context):
        return f"""<s>[INST] You are an expert FAQ assistant. Use this context to answer:
        
        {context}
        
        Question: {question} 
        Answer: [/INST]"""

    def get_answer(self, question):
        if not self.model:
            return "System initializing, please wait..."
            
        context = "\n".join(self.retriever.retrieve(question))
        prompt = self.format_prompt(question, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            repetition_penalty=1.1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)