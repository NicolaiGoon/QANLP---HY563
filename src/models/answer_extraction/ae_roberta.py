from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

class ae_roberta:
    '''
    https://huggingface.co/deepset/roberta-base-squad2
    '''
    
    def __init__(self):
        model_name = "deepset/roberta-base-squad2"
        self.nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    def predict(self,q,text):
        return self.nlp({'question':q,'context':text})['answer']