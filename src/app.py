'''
this script starts the evaluation
'''

from collection import collection
from metrics import metrics
from models.answer_extraction.ae_minilm import ae_minilm
from models.answer_extraction.ae_distilbert import ae_distilbert
from models.answer_extraction.ae_roberta import ae_roberta 
from models.confirmation.cq_minilm import cq_minilm
from models.confirmation.cq_qnli import cq_qlni
from models.answer_extraction.ae_expansion import ae_expansion
import time
import json

def evaluate(pred,gold,label=""):
    em = metrics.exactMatch(pred,gold)
    f1 = metrics.f1score(pred,gold)
    print("\n"+label)
    print("EM: "+str(em))
    print("F1: "+str(f1))
    return em,f1

outfile = {}

print("creating collection ...")
c = collection()
print("collection ok")
print("Evaluating Baseline (Returns random answers):")
pred,gold = c.getRandomAnswers()
em,f1 = evaluate(pred['factoid'],gold['factoid'])
outfile['BaseLine'] = []
outfile['BaseLine'].append(em)
outfile['BaseLine'].append(f1)
print("Evaluating Neural Models:")

models = [ae_minilm(),ae_roberta(),ae_distilbert(),ae_expansion(),ae_expansion(option="concat")]
model_names = ["minilm","roberta","distilbert","roberta_exp_comment","roberta_exp_concat"]

for model,name in zip(models,model_names):
    outfile[name] = []
    print("Start Evaluating "+name)
    start = time.time()
    results = {'factoid':[],'definition':[],'confirmation':[],'CARDINAL':[],'PERSON':[],'DATE-TIME':[]}
    gold = {'factoid':[],'definition':[],'confirmation':[],'CARDINAL':[],'PERSON':[],'DATE-TIME':[]}
    topic_count = 1
    total_topics = len(c.topics) 
    for topic in c.topics:
        text = topic["text"]
        print("Topic "+str(topic_count)+'/'+str(total_topics)+" : "+topic['title'])
        qcount = 1
        qtotal = len(topic["qa"])
        for q in topic["qa"]:
            question = q["question"]
            print("Question "+str(qcount)+"/"+str(qtotal)+" : "+question)
            answer = model.predict(question,text)
            print("Answer: "+answer)
            print()
            results[q["type"]].append(answer)
            if(q["entity"] in ["PERSON","DATE-TIME","CARDINAL"]):
                results[q["entity"]].append(answer)
                gold[q["entity"]].append(q["answer"])
            gold[q["type"]].append(q["answer"])
            qcount+=1
        topic_count+=1
    end = time.time()
    em,f1 = evaluate(results['factoid'],gold['factoid'],label="factoid")
    outfile[name].append((em,f1))
    em,f1 = evaluate(results['definition'],gold['definition'],label="definition")
    outfile[name].append((em,f1))
    em,f1 = evaluate(results['CARDINAL'],gold['CARDINAL'],label="cardinal")
    outfile[name].append((em,f1))
    em,f1 = evaluate(results['PERSON'],gold['PERSON'],label="person")
    outfile[name].append((em,f1))
    em,f1 = evaluate(results['DATE-TIME'],gold['DATE-TIME'],label="Date-time")
    outfile[name].append((em,f1))
    em,f1 = evaluate(results['factoid']+results['definition'],gold['factoid']+gold['definition'],label="Average")
    outfile[name].append((em,f1))
    outfile[name].append({"time" : end-start})
    print("total time: "+str(end-start))


print("Evaluating Confirmation")
cq_models = [cq_minilm(),cq_qlni()]
cq_names = ["embeddings","cross-encoder"]
for model,name in zip(cq_models,cq_names):
    outfile[name] = []
    results = []
    gold = []
    start = time.time()
    for topic in c.topics:
        for q in topic["qa"]:
            text = topic["text"]
            if(q["type"] == "confirmation"):
                gold.append(q["answer"])
                question = q["question"]
                if(name == "embeddings"):
                    answer = models[1].predict(question,text)
                    score = model.getSimilarity(answer,text)
                else: 
                    score = model.getSimilarity(question,text)
                if score > 0.7 : results.append('Yes')
                else : results.append('No')
    end = time.time()
    em,_ = evaluate(results,gold,label="Confirmation")
    outfile[name].append(em)
    outfile[name].append({"time" : end-start})  

model = cq_models[1]
texts = [x["text"] for x in c.topics]
accuracy = 0
counter = 1
outfile["cross-encoder-passage"] = []
for topic in c.topics:
    for q in topic["qa"]:
        question = q["question"]
        print(question)
        correct_text = topic["text"]
        results = model.getSimilarities(question,texts)
        if(correct_text == results[0][0]): accuracy += 1
        counter += 1
accuracy /= counter
outfile["cross-encoder-passage"].append(accuracy)


with open("results.json","w") as f:
    json.dump(outfile,f)
