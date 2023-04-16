# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:58:20 2023

@author: Malay 
"""

import pickle 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
page_bg_img = """
<style>
.stApp {
        background: rgb(20,21,25);
        background: linear-gradient(90deg, rgba(0,0,0,1) 10%, rgba(35,36,43,0.17) 61%, rgba(110,114,142,1) 100%);
    }
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
#loading the saved models
parkinson_model = pickle.load(open('parkinsons_model1.sav', 'rb'))
parkinson_model_mutual = pickle.load(open('parkinsons_model_mutual1.sav', 'rb'))

parkinsons_data = pd.read_csv('Parkinsson disease (1).csv')
parkinsons_data_mutual = pd.read_csv('Parkinsson disease (2).csv')

X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']

X_2 = parkinsons_data_mutual.drop(columns=['name','status','MDVP:Jitter(%)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','Shimmer:APQ3','Shimmer:DDA','NHR','RPDE','DFA','D2'], axis=1)
Y_2 = parkinsons_data_mutual['status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2, stratify=Y)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_2, Y_2, test_size=0.2, random_state=2, stratify=Y_2)

scaler = StandardScaler()
scaler2 = StandardScaler()

scaler.fit(X_train)
scaler2.fit(X_train2)


def validate_input(input_string):
    # Check if input is not empty
    if not input_string:
        return False
    try:
        float(input_string)
        return True
    except ValueError:
        return False


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Parkinsons Disease Prediction System',
                          
                          ['Full Feature Prediction',
                           'Selected Feature Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)
    
    
# Parkinson's Prediction Page Full features
if (selected == "Full Feature Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using Full features")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP RAP')
        
    with col2:
        PPQ = st.text_input('MDVP PPQ')
        
    with col3:
        DDP = st.text_input('Jitter DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer APQ5')
        
    with col3:
        APQ = st.text_input('MDVP APQ')
        
    with col4:
        DDA = st.text_input('Shimmer DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    fo_float = float(fo) if fo != '' else None
    fhi_float = float(fhi) if fhi != '' else None
    flo_float = float(flo) if flo != '' else None
    Jitter_percent_float = float(Jitter_percent) if Jitter_percent != '' else None
    Jitter_Abs_float = float(Jitter_Abs) if Jitter_Abs != '' else None
    RAP_float = float(RAP) if RAP != '' else None
    PPQ_float= float(PPQ) if PPQ != '' else None
    DDP_float = float(DDP) if DDP != '' else None
    Shimmer_float = float(Shimmer) if Shimmer != '' else None
    Shimmer_dB_float = float(Shimmer_dB) if Shimmer_dB != '' else None
    APQ3_float = float(APQ3) if APQ3 != '' else None
    APQ5_float = float(APQ5) if APQ5 != '' else None
    APQ_float = float(APQ) if APQ != '' else None
    DDA_float= float(DDA) if DDA != '' else None
    NHR_float = float(NHR) if NHR != '' else None
    HNR_float = float(HNR) if HNR != '' else None
    RPDE_float = float(RPDE) if RPDE != '' else None
    DFA_float = float(DFA) if DFA != '' else None
    spread1_float = float(spread1) if spread1 != '' else None
    spread2_float = float(spread2) if spread2 != '' else None
    D2_float = float(D2) if D2 != '' else None
    PPE_float = float(PPE) if PPE != '' else None
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    
    input_data = pd.DataFrame({
        'MDVP:Fo(Hz)': [fo_float],
        'MDVP:Fhi(Hz)': [fhi_float],
        'MDVP:Flo(Hz)': [flo_float],
        'MDVP:Jitter(%)': [Jitter_percent_float],
        'MDVP:Jitter(Abs)': [Jitter_Abs_float],
        'MDVP:RAP': [RAP_float],
        'MDVP:PPQ': [PPQ_float],
        'Jitter:DDP': [DDP_float],
        'MDVP:Shimmer': [Shimmer_float],
        'MDVP:Shimmer(dB)': [Shimmer_dB_float],
        'Shimmer:APQ3': [APQ3_float],
        'Shimmer:APQ5': [APQ5_float],
        'MDVP:APQ': [APQ_float],
        'Shimmer:DDA':[DDA_float],
        'NHR': [NHR_float],
        'HNR': [HNR_float],
        'RPDE': [RPDE_float],
        'DFA': [DFA_float],
        'spread1': [spread1_float],
        'spread2': [spread2_float],
        'D2': [D2_float],
        'PPE': [PPE_float]
    })
    
    input_data_scaled = scaler.transform(input_data)
    print(input_data_scaled)
    # creating a button for Prediction    
    if st.button("Parkinson's Result Full"):
        
        if validate_input(fo) and validate_input(fhi) and validate_input(flo) and validate_input(Jitter_percent) and validate_input(Jitter_Abs) and validate_input(RAP) and validate_input(PPQ) and validate_input(DDP) and validate_input(Shimmer) and validate_input(Shimmer_dB) and validate_input(APQ3) and validate_input(APQ5) and validate_input(APQ) and validate_input(DDA) and validate_input(NHR) and validate_input(HNR) and validate_input(RPDE) and validate_input(DFA) and validate_input(spread1) and validate_input(spread2) and validate_input(D2) and validate_input(PPE):
            parkinsons_prediction = parkinson_model.predict(input_data_scaled)                          
        
            if (parkinsons_prediction[0] == 1):
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
            st.success(parkinsons_diagnosis)
        
        else:
            st.error("Invalid inputs. Please enter numbers only.")
            
            
        


    

# Parkinson's Prediction Page Selected features 
if (selected == "Selected Feature Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using Selected Features")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo1 = st.text_input('MDVP Fo(Hz)')
        
    with col2:
        fhi1 = st.text_input('MDVP Fhi(Hz)')
        
    with col3:
        flo1 = st.text_input('MDVP Flo(Hz)')
               
    with col4:
        Jitter_Abs1 = st.text_input('MDVP Jitter(Abs)')
        
    with col5:
        Shimmer_dB1 = st.text_input('MDVP Shimmer(dB)')
        
    with col1:
        APQ51 = st.text_input('Shimmer APQ5')
               
    with col2:
        APQ1 = st.text_input('MDVP APQ')
        
    with col3:
        HNR1 = st.text_input('HNR')
             
    with col4:
        spread11 = st.text_input('spread1')
        
    with col5:
        spread21 = st.text_input('spread2')
               
    with col1:
        PPE1 = st.text_input('PPE')
        
    fo1_float = float(fo1) if fo1 != '' else None
    fhi1_float = float(fhi1) if fhi1 != '' else None
    flo1_float = float(flo1) if flo1 != '' else None
    Jitter_Abs1_float = float(Jitter_Abs1) if Jitter_Abs1 != '' else None
    Shimmer_dB1_float = float(Shimmer_dB1) if Shimmer_dB1 != '' else None
    APQ51_float = float(APQ51) if APQ51 != '' else None
    APQ1_float = float(APQ1) if APQ1 != '' else None
    HNR1_float = float(HNR1) if HNR1 != '' else None
    spread11_float = float(spread11) if spread11 != '' else None
    spread21_float = float(spread21) if spread21 != '' else None
    PPE1_float = float(PPE1) if PPE1 != '' else None
    
    # code for Prediction
    parkinsons_diagnosis_mutual = ''
    
    input_data_mutual = pd.DataFrame({
        'MDVP:Fo(Hz)': [fo1_float],
        'MDVP:Fhi(Hz)': [fhi1_float],
        'MDVP:Flo(Hz)': [flo1_float],
        'MDVP:Jitter(Abs)': [Jitter_Abs1_float],
        'MDVP:Shimmer(dB)': [Shimmer_dB1_float],
        'Shimmer:APQ5': [APQ51_float],
        'MDVP:APQ': [APQ1_float],
        'HNR': [HNR1_float],
        'spread1': [spread11_float],
        'spread2': [spread21_float],
        'PPE': [PPE1_float]
    })
    input_data_scaled_mutual = scaler2.transform(input_data_mutual)
    # creating a button for Prediction    
    if st.button("Parkinson's Result Selected"):
        if validate_input(fo1) and validate_input(fhi1) and validate_input(flo1) and validate_input(Jitter_Abs1) and validate_input(Shimmer_dB1) and validate_input(APQ51) and validate_input(APQ1) and validate_input(HNR1) and validate_input(spread11) and validate_input(spread21) and validate_input(PPE1):
            parkinsons_prediction_mutual = parkinson_model_mutual.predict(input_data_scaled_mutual)                          
        
            if (parkinsons_prediction_mutual[0] == 1):
                parkinsons_diagnosis_mutual = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis_mutual = "The person does not have Parkinson's disease"
        
            st.success(parkinsons_diagnosis_mutual)
            
        else:
            st.error("Invalid inputs. Please enter numbers only.")
            
            
