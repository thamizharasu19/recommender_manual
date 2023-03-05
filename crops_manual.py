import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
#rf - 99
#ml - 95
#nb - 87
#sg - 70
#gp - 65
st.set_page_config(
    page_title="CROP SUGGESTER",
    page_icon="graph",
    layout="centered",
)
st.title("🍃CROP SUGGESTER SYSTEM🍃")

model_rf = joblib.load('Crop_rf')
model_ml = joblib.load('Crop_ml')
model_nb = joblib.load('Crop_nb')
model_sd = joblib.load('Crop_sd')


n = st.number_input("ENTER THE NITROGEN LEVEL IN THE SOIL : ")
p = st.number_input("ENTER THE PHOSPHOROUS LEVEL IN THE SOIL : ")
k = st.number_input("ENTER THE POTTASIUM LEVEL IN THE SOIL : ")
t = st.number_input("ENTER THE TEMPARATURE LEVEL ")
h = st.number_input("ENTER THE HUMIDITY LEVEL IN THE SOIL : ")
ph = st.number_input("ENTER THE PH LEVEL IN THE SOIL : ")
r = st.number_input("ENTER THE RAINFALL LEVEL : ")


output = {

    'rice' : 1,
    'maize' : 2,
    'chickpea' : 3,
    'kidneybeans':4,
    'pigeonpeas' : 5,
    'mothbeans' : 6,
    'mungbean' : 7,
    'blackgram' : 8,
    'lentil' : 9,
    'pomegranate' : 10,
    'banana' : 11,
    'mango' : 12,
    'grapes' : 13,
    'watermelon' : 14,
    'muskmelon' : 15,
    'apple' : 16,
    'orange' : 17,
    'papaya' : 18,
    'coconut': 19,
    'cotton' : 20,
    'jute' : 21,
    'coffee' : 22
}

key_out = list(output.keys())
val_out = list(output.values())

def crop_suggest():
    rows = np.array([n,p,k,t,h,ph,r])
    X = pd.DataFrame([rows])
    pred_rf = model_rf.predict(X)[0]
    pred_ml = model_ml.predict(X)[0]
    pred_nb = model_nb.predict(X)[0]
    pred_sd = model_sd.predict(X)[0]
    res = [pred_rf,pred_ml,pred_nb,pred_sd]
    s = [set(res)]
    for i in range(len(s)):
        pos = val_out.index(s[i])
        st.info('THE BEST CROP FOR YOU IS: %s' %(key_out[pos]))

st.button('PREDICT', on_click = crop_suggest)
