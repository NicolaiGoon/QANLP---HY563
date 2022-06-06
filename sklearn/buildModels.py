from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

# Train the models with all the data and export them

edf = pd.read_csv('./sklearn/entities.csv')
qdf = pd.read_csv('./sklearn/questions.csv')

X_e = edf.loc[:,edf.columns != 'class']
Y_e = edf['class']

X_q = qdf.loc[:,qdf.columns != 'class']
Y_q = qdf['class']

qtp =  RandomForestClassifier()
etp = SVC()

qtp.fit(X_q,Y_q)
etp.fit(X_e,Y_e)

pickle.dump(qtp,open('./sklearn/qtp_model.sav','wb'))
pickle.dump(etp,open('./sklearn/etp_model.sav','wb'))