import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report,confusion_matrix,mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import streamlit as st
import re
st.set_page_config(
    page_title="INDUSTRIAL COPPER MODELING",
    layout="wide",
    initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .header-container {
        background-color: red; /* Purple background */
        padding: 20px; /* Increased padding */
        border-radius: 15px; /* Rounded corners */
        width: 100%; /* Make the container broader */
        margin-left: 10px; /* Align to left margin */
        margin-bottom: 20px; /* Reduce gap between header and subheader */
    }
    .header-text {
        font-family: Arial, sans-serif;
        color: white; /* White text color */
        margin: 0; /* Remove default margin */
        font-size: 24px; /* Increase font size */
    }
    .subheader-container {
        background-color: purple; /* Purple background */
        padding: 5px; /* Padding around the text */
        border-radius: 10px; /* Rounded corners */
        width: 100%; /* Width of the container */
        margin-left: 10px; /* Align to left margin */
        margin-top: -10px; /* Reduce gap between header and subheader */
    }
    .subheader-text {
        font-family: Arial, sans-serif;
        color: white; /* White text color */
        font-size: 24px; /* Font size */
        margin: 0; /* Remove default margin */
    }
    </style>
    """,
    unsafe_allow_html=True
)    

# Display the header with custom styling
st.markdown('<div class="header-container"><p class="header-text">Industrial Copper Modeling</p></div>', unsafe_allow_html=True)

# Display the subheader with custom styling
st.markdown('<div class="subheader-container"><p class="subheader-text">Status and Selling price prediction Application</p></div>', unsafe_allow_html=True)
with open("rfclas.pkl","rb") as clasr:
    rf_model_new=pickle.load(clasr)
with open("rfreg.pkl","rb") as rregr:
    rr_model_new=pickle.load(rregr)
#with open("scal.pkl","rb") as ssr:
    #scaler_new=pickle.load(ssr)
with open("lab.pkl","rb") as lbr:
    lab_new=pickle.load(lbr)
with st.container(height=700,border=True):
    st.image('C:/Users/ABI/Desktop/project5/copp.jpg',caption='Industrial Copper Modeling')


tab1,tab2=st.tabs(["PREDICT STATUS","PREDICT SELLING PRICE"])

country_code=[78.0,26.0,25.0,27.0,32.0,28.0,84.0,77.0,30.0,39.0,79.0,38.0,40.0,80.0,113.0,89.0,107.0]
status_code=['Won','Lost','Not lost for AM','Revised','To be approved','Draft','Offered','Offerable','Wonderful']
status_d={'Lost':0, 'Won':1, 'Draft':2, 'To be approved':3, 'Not lost for AM':4,
                'Wonderful':5, 'Revised':6, 'Offered':7, 'Offerable':8}
item_code=[ "W",'S','PL','Others','WI','IPL','SLAWR']
item_code_d={'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}
application_code=[10.0,41.0,15.0,59.0,42.0,56.0,29.0,27.0,26.0,28.0,40.0,25.0,79.0,22.0,20.0,66.0,3.0,38.0,58.0,4.0,39.0,65.0,67.0,68.0,99.0,19.0,69.0,5.0,70.0,2.0]
productreference_code=[611993,164141591,640665,1670798778,628377,1668701718,640405,1671863738,1332077137,1693867550,1668701376,1671876026,628117,164337175,1668701698,1693867563,1282007633,1721130331,1665572374,628112,611728,1690738206,1722207579,640400,1668701725,164336407,611733,1690738219,1665584320,1665572032,1665584642,929423819,1665584662]
with tab1:
    st.write("Predict Status")
    # Remove trailing whitespaces from each item code
    item_code_stripped = [item.strip() for item in item_code]
    country=st.selectbox("Country",country_code)
    item_type=st.selectbox("Item",item_code_stripped)
    application=st.selectbox("Application",application_code)
    product_ref=st.selectbox("Product Reference",productreference_code)
    
    quantity_tons=st.text_input("Quantity in Tons min_valu e=0.00001,max_value=1000000000.0")
    customer=st.text_input("Customer N min_value=12458.0,max_value=30408185.0")
    thickness=st.text_input("Thickness min_value=0.18,max_value=400.0")
    width=st.text_input("Width min_value=1.0,max_value=2990.0")
    selling_price=st.text_input("Selling Price,min_value=0.1,max_value=100001015.0")
    
    product_ref = int(product_ref)
   
    
    cflag=0 
    pattern = "^(?:\d+|\d*\.\d+)$"
    for k in [quantity_tons,thickness,width,customer,selling_price]:             
        if re.match(pattern, k):
            pass
        else:                    
            cflag=1  
            break
            
    if cflag==1:
        if len(k)==0:
            st.write("please enter a valid number space not allowed")
        else:
                st.write("You have entered an invalid value: ",k)   
    if cflag==0:
        if st.button('Predict Status'):
                        
            #sample=np.array([[np.log(float(quantity_tons)),int(customer),country,application,np.log(float(thickness)),float(width),int(product_ref),np.log(float(selling_price)),item_code_d[item_type]]])
            columns = ['quantity_tons', 'customer', 'country', 'item_type','application', 'thickness', 'width', 'product_ref', 'selling_price' ]
            sample_data = {
            'quantity_tons': [np.log(float(quantity_tons))],
            'customer': [int(customer)],
            'country': [country],
            'item_type': [item_code_d[item_type.strip()]],
            'application': [application],
            'thickness': [np.log(float(thickness))],
            'width': [float(width)],
            'product_ref': [product_ref],
            'selling_price': [np.log(float(selling_price))]
            }
            sample_df = pd.DataFrame(sample_data, columns=columns)
            predict_status=rf_model_new.predict(sample_df)
            if(predict_status==1):
               st.write("Won")
            else:
               st.write("Lost")
with tab2:
    st.write("Predict Selling Price")
    #item_code_stripped = [item.strip() for item in item_code]
    country=st.selectbox("Country",country_code,key="51")
    item_type=st.selectbox("Item",item_code,key="52")
    application=st.selectbox("Application",application_code,key="53")
    product_ref=st.selectbox("Product Reference",productreference_code,key="54")
    status=st.selectbox("Status",status_code,key="55")
    quantity_tons=st.text_input("Quantity in Tons min_valu e=0.00001,max_value=1000000000.0",key="56")
    customer=st.text_input("Customer N min_value=12458.0,max_value=30408185.0",key="57")
    thickness=st.text_input("Thickness min_value=0.18,max_value=400.0",key="58")
    width=st.text_input("Width min_value=1.0,max_value=2990.0",key="59")
    
    
    if st.button('Predict Price'):
        #sample_input=np.array([[np.log(float(quantity_tons)),int(customer),country,application,np.log(float(thickness)),float(width),int(product_ref),status_d[status],item_code_d[item_type]]])
        columns = ['quantity_tons', 'customer', 'country', 'status','item_type','application', 'thickness', 'width', 'product_ref'  ]
        sample_data = {
            'quantity_tons': [np.log(float(quantity_tons))],
            'customer': [int(customer)],
            'country': [country],
            'status': [status_d[status]],
            'item_type': [item_code_d[item_type.strip()]],
            'application': [application],
            'thickness': [np.log(float(thickness))],
            'width': [float(width)],
            'product_ref': [product_ref],
            
            }
        sample_df = pd.DataFrame(sample_data, columns=columns)
        predict_sp=rr_model_new.predict(sample_df)
        selling_price=np.exp(predict_sp[0])
        selling_price=round(selling_price,2)
        st.write("Selling Price: ",selling_price)
    
