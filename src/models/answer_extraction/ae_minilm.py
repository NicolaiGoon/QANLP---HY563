from transformers import AutoModelForQuestionAnswering,  AutoTokenizer, pipeline

class ae_minilm:
    '''
    https://huggingface.co/deepset/minilm-uncased-squad2
    '''
    def __init__(self):
        model_name = "deepset/minilm-uncased-squad2"
        self.nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    def predict(self,q,text):
        return self.nlp({'question':q,'context':text})['answer']