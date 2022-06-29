from html import entities
from models.answer_extraction.ae_roberta import ae_roberta
from models.answer_extraction.ner import getEntities
from models.answer_extraction.entity_expansion import getComment

class ae_expansion:
    """
    uses entity expansion
    """

    def __init__(self,option=None):
        self.model = ae_roberta()
        self.option = option

    def  predict(self,q,text):
        entities = getEntities(q)
        if(len(entities) > 0):
            comment = getComment(entities[0])
            if(comment):
                # concat text with comment
                if(self.option == "concat"):
                    return self.model.predict(q,text) + " " + self.model.predict(q,comment)
                # use only the comment for passage
                else:
                    return self.model.predict(q,comment)
        # no comment or no entity found
        return self.model.predict(q,text)


