from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

class ae_distilbert:
    '''
    https://huggingface.co/deepset/minilm-uncased-squad2
    '''
    def __init__(self):
        model_name = "distilbert-base-cased-distilled-squad"
        self.nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    def predict(self,q,text):
        return self.nlp({'question':q,'context':text})['answer']

m = ae_distilbert()
print(m.predict("what is nikos?","nikos is a pirate"))