#from pyexpat import features
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np

st.write("""
# creditworthiness of potential customers Prediction App This app predicts the **creditworthiness **
""")


st.sidebar.header('User Input Parameters')



def user_input_features():
   SEGMENT = st.sidebar.slider('SEGMENT', 10, 10, 90)
   TYPE = st.sidebar.slider('TYPE', 0, 1, 1)
   Remise = st.sidebar.slider('Remise', 0, 1, 13)
   Echeance = st.sidebar.slider('Echeance', 0, 5, 90)
   DMA = st.sidebar.slider('DMA', 0, 1, 800000)
   REGION = st.sidebar.slider('REGION', 1010, 10, 1170)
   data = {'SEGMENT': SEGMENT,
           'TYPE': TYPE,
           'Remise': Remise,
           'Echeance': Echeance,
           'DMA': DMA,
           'REGION': REGION}
   features = pd.DataFrame(data, index=[0])
   return features

# def user_input_features():
#     #SEGMENT = st.sidebar.slider('SEGMENT', 10, 10, 90)
#     SEGMENT = st.selectbox('Inserez SEGMENT',('AGRICULTURE','BTP','CARRIERE','COMMERCE','INDUSTRIE','SERVICE','TEXTILE','TOURISME','TRANSPORT'))
#     TYPE = st.sidebar.slider('TYPE', 0, 1, 1)
#     Remise = st.sidebar.slider('Remise', 0, 1, 13)
#     Echeance = st.sidebar.slider('Echeance', 0, 5, 90)
#     DMA = st.sidebar.slider('DMA', 0, 1, 800000)
#     REGION = st.sidebar.slider('REGION', 1010, 10, 1170)
#     data = {'SEGMENT': SEGMENT,
#             'TYPE': TYPE,
#             'Remise': Remise,
#             'Echeance': Echeance,
#             'DMA': DMA,
#             'REGION': REGION}
#     features = pd.DataFrame(data, index=[0])
#     return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)


data = pd.read_excel('TabModel2.xlsx')
data.head()

data['SEGMENT'].replace(['AGRICULTURE','BTP','CARRIERE','COMMERCE','INDUSTRIE','SERVICE','TEXTILE','TOURISME','TRANSPORT'],[10,20,30,40,50,60,70,80,90],inplace=True)
data['TYPE'].replace(['VRAC','MAD'],[0,1],inplace=True)
data['REGION'].replace(['ARIANA','BEN AROUS','BIZERTE','GABES','GAFSA','KAIROUAN','KEBILI','MAHDIA','MONASTIR','NABEUL','SFAX','SIDI BOUZID','SILIANA','SOUSSE','TOZEUR','TUNIS','ZAGHOUAN'],[1010,1020,1030,1040,1050,1060,1070,1080,1090,1100,1110,1120,1130,1140,1150,1160,1170],inplace=True)
y = data['Solvabilite']
x = data.drop('Solvabilite', axis=1)
#data.head()
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.1)
clf = SGDClassifier(loss="log", penalty="l2")
clf.fit(trainX, trainY)

y_pred = clf.predict(testX)

#print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
#def solv(clf, SEGMENT=90, TYPE=1, Remise=5, Echeance=0, DMA=5000, REGION=1140):
#    r = np.array([SEGMENT, TYPE, Remise, Echeance, DMA, REGION]).reshape(1, 6)
#    print(clf.predict(r))

#solv(clf)
#clf.predict_proba(testX)

#st.subheader('Class labels and their corresponding index number')
#st.write(y)

st.subheader('Prediction')

y = clf.predict(df)
st.write(y)
#st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
#st.write(prediction_proba)
y2 = clf.predict_proba(df)
st.write(y2)
st.write(y2[0])
st.write(y2[0][0])
st.write(y2[0][1])