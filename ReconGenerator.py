from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import filetype
import pickle
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

with st.sidebar:
    add_radio = st.selectbox("Options",
                         ("Reconciliation Manager", "Budgeting and Forecasting")
                         )


@st.cache_resource
def get_recon_base_model():
    recon_base_model = pickle.load(open('xgb_recon_regress.pkl', "rb"))
    return recon_base_model

recon_clf = get_recon_base_model()


st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def get_dep_data():
    actuals = pd.read_csv('departmentspend.csv')
    budget_prediction = pd.read_csv('budgetprediction.csv')
    budget_prediction[['YEAR','MONTH']] = budget_prediction['Date'].str.split('-',expand=True)
    annual_budg_pred = budget_prediction.groupby(['YEAR','Department'], as_index=False)[['Amount']].sum()
    annual_budg_pred['YEAR'] = annual_budg_pred['YEAR'].astype(int)
    annual_budg_pred.loc[annual_budg_pred['YEAR'] <= 2023, 'Amount'] = 0
    return actuals, annual_budg_pred

actuals, annual_budg_pred = get_dep_data()

@st.cache_resource
def get_department_options(spend_df):
    departments = spend_df['Department'].unique()
    department_list = departments.tolist()
    department_list = sorted(department_list) 
    return department_list  

department_list = get_department_options(annual_budg_pred)


#st.title()
def file_type_detection(file):
    kind = filetype.guess(file)
    if kind is None:
        return "CSV successfully uploaded."
    return kind.extension




def preprocessing(recon_data):
    # Encoding all the Variables for Recon Analysis
    #label_encoder = LabelEncoder()

    # Encoding the Department Variable
    recon_data['Department'] = recon_data['Department'].astype('category')
    #recon_data['Department_Cat'] = recon_data['Department'].cat.codes

    # Encoding the Program variable
    recon_data['Program'] = recon_data['Program'].astype('category')
    #recon_data['Program_Cat'] = recon_data['Program'].cat.codes

    # Encoding the Expense Category variable
    recon_data['Expense Category'] = recon_data['Expense Category'].astype('category')
    #recon_data['Expense_Category_Cat'] = recon_data['Expense Category'].cat.codes

    # Encoding the Fund variable
    recon_data['Fund'] = recon_data['Fund'].astype('category')
    #recon_data['Fund_Cat'] = recon_data['Fund'].cat.codes

    # Encoding the Fund Type variable
    recon_data['Fund Type'] = recon_data['Fund Type'].astype('category')
    #recon_data['Fund_Type_Cat'] = recon_data['Fund Type'].cat.codes

    # Encoding the Vendor variable
    recon_data['Vendor'] = recon_data['Vendor'].astype('category')
    #recon_data['Vendor_Cat'] = recon_data['Vendor'].cat.codes

    # Encoding the Payment Status variable
    #recon_data['Payment Status'] = recon_data['Payment Status'].astype('category')
    #recon_data['Payment_Status_Cat'] = recon_data['Payment Status'].cat.codes
    #recon_data.loc[recon_data["Payment Status"] == 'Void-Reconciled', "Payment Status"] = "Paid-Reconciled" # Doing this for model simplicity in demo 
    #recon_data['Payment Status'] = label_encoder.fit_transform(recon_data['Payment Status'])

    # Encoding the Vendor Type variable
    recon_data['Vendor Type'] = recon_data['Vendor Type'].astype('category')
    #recon_data['Vendor_Type_Cat'] = recon_data['Vendor Type'].cat.codes

    # Encoding the Payment Method variable 
    recon_data['Payment Method'] = recon_data['Payment Method'].astype('category')
    #recon_data['Payment_Method_Cat'] = recon_data['Payment Method'].cat.codes

    # Encoding the Service variable
    recon_data['Service'] = recon_data['Service'].astype('category')
    #recon_data['Service_Cat'] = recon_data['Service'].cat.codes
    Recon_Test_X = recon_data.drop(['Payment Status'], axis=1)
    Recon_Test_y = recon_data['Payment Status']
    return Recon_Test_X, Recon_Test_y





def run_model(recon_data):
    Recon_Test_X, Recon_Test_y = preprocessing(recon_data)
#    st.write(Recon_Test_X.dtypes)
    recon_results = recon_clf.predict(Recon_Test_X)
    st.subheader("Reconciliation Results")
    st.write("The total number of transactions entered and processed is ", str(len(recon_results)), ", review the additional output details and download transactions below.")
    st.markdown("**Total Predicted as Reconciled**")
    st.write(str(np.count_nonzero(recon_results==0)), " transactions are projected to be reconciled.")
    st.write(str(np.count_nonzero(recon_results==1)), " transactions are projected to be unreconciled.")
    full_recon_results = pd.concat([Recon_Test_X, pd.DataFrame(recon_results, columns =['Predictions'], index=Recon_Test_X.index)], axis=1)
    export_df = full_recon_results.loc[full_recon_results['Predictions'] == 1]
    #full_recon_results['Actuals'] = Recon_Test_y
    #st.write(full_recon_results.loc[full_recon_results['Predictions'] == 0])
    c_clf = confusion_matrix(np.array(Recon_Test_y).astype(float), np.array(recon_results.astype(float)))
    #st.write(c_clf)
    st.write("The overall accuracy for this set of transactions is: ", str(accuracy_score(Recon_Test_y, recon_results)*100),"%.")
    cf_data = {
        'Result Status' : ['Correctly Predicted Reconcilation of Transaction','Correctly Predicted Unreconciled Transaction', 'Predicted Unreconciled but was Reconciled', 'Predicted Reconciled but was Unreconciled'],
        'Number Identified' : [c_clf[0][0], c_clf[1][1], c_clf[1][0], c_clf[0][1]] 
    }
    cf_df = pd.DataFrame(cf_data)
    st.markdown(cf_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    st.markdown("")

    # Get the CSV prepared for when the button is clicked
    
    export_csv = export_df.to_csv(index=False)

    btn = st.download_button('Download the CSV of Unreconciled Transactions', data=export_csv, file_name="unrecon_transactions.csv", mime='text/csv')


def file_upload():
    file = st.file_uploader("Upload Transactions", type=[
        'xlsx','csv','xls','xlsm','xlsb','odf','ods','odt'
        ])
    if file:
        type = file_type_detection(file)
        st.write(type)
        if type == 'csv':
            recon_data = pd.read_csv(file)
            run_model(recon_data)
        else:
            recon_data = pd.read_csv(file)
            run_model(recon_data)
#            usecases 
        return recon_data
    
def display_model_info():
    
    variable_desc = {
        'Variable' : ['Fiscal Year Period', 'Service','Deparment', 'Program', 'Account', 'Fund Code', 'Fund', 'Fund Type','Vendor','Payment Method','Amount','Vendor Type', 'Existing Vendor Relationship', 'Internal Operations', 'Transaction Dates'],
        'Description' : ['Accounting Period (Month).', 'Type of Service Performed.', 'Which Government Department this transaction was recorded in.', 'Which Program this transaction was recorded in.', 'The GL account that it was recorded in.',
                         'The Fund it was recorded in.', 'A description of that fund.', 'A description of the type of Fund.', 'Internal or external vendor that received the payment.', 'What payment method was used (Check, ACH, Wire, Giro, etc.).',
                         'Transaction amount.','LLC, Governmental, Individual, etc.', 'If this is an existing relationship in previous periods or a new one.', 'If this transaction was Government to Government or what would amount to "intercompany."',
                         'Day of the Month and day of the week this transaction was entered.']
    }
    variable_desc_df = pd.DataFrame(variable_desc)
       
    st.subheader('Model Performance on Reserved Test Data Set')
    st.markdown('The overall accuracy of this model showed that it performed **162%** better than a random classifier, even with the limited feature engineering used.')
    cf_data = {
        'Result Status' : ['Correctly Predicted Reconcilation of Transaction','Correctly Predicted Unreconciled Transaction', 'Predicted Unreconciled but was Reconciled', 'Predicted Reconciled but was Unreconciled'],
        'Number Identified' : ['1,288,348', '442,326','174,268', '138,900'] 
    }
    cf_df = pd.DataFrame(cf_data)
    st.markdown(cf_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)


    st.subheader('Target Variable')
    st.markdown('The target variable for this analysis with *reconciliation status*, which would enable prediction of whether a single sided entry would be reconciled at close.')
    
    st.subheader("Variables Used for Analysis")
    st.markdown('This table offers a short description of the variables used in building this ML model.')
    st.markdown(variable_desc_df.style.hide(axis="index").to_html(), unsafe_allow_html=True) 


def plot_budget_projection(department):
    in_var_df = actuals.loc[actuals['Department'] == department]
    pred_df = annual_budg_pred.loc[annual_budg_pred['Department'] == department]


    pred_df['Amount'].loc[pred_df['YEAR'] == 2024] = round(annual_budg_pred['Amount'].loc[(annual_budg_pred['YEAR'] == 2024) & (annual_budg_pred['Department'] == department)].tolist()[0] - actuals['Amount'].loc[(actuals['Fiscal Year'] == 2024) & (actuals['Department'] == department)].tolist()[0])


    new_year = {'Fiscal Year': 2025, 'Department': department, 'Amount': 0}
    new_year_df = pd.DataFrame(new_year, index=[0])
    in_var_df2 = pd.concat([in_var_df, new_year_df], ignore_index=True)



    plt.bar(in_var_df2['Fiscal Year'], in_var_df2['Amount'], color='slategrey')
    plt.bar(pred_df['YEAR'], pred_df['Amount'], bottom=in_var_df2['Amount'], color = 'dodgerblue')    

    plt.legend(['Actuals','Forecasted'])
    plt.title(department)
    plt.yticks([])
    plt.show()
    fig = plt.show()
    st.pyplot(fig)

    pred_df_2 = pred_df.loc[pred_df['YEAR'] > 2023]
    pred_df_2 = pred_df_2.rename(columns={'YEAR':'Fiscal Year'})

    st.markdown('The table below outlines the values associated with the chart above. Pleasr review for detailed information')
    st.subheader('Actuals:')
    st.markdown(in_var_df2.style.format({"Amount": "{:.2f}"}).hide(axis="index").to_html(), unsafe_allow_html=True)
    st.subheader('Projected Additional Spend:')
    st.markdown(pred_df_2.style.format({"Amount": "{:.2f}"}).hide(axis="index").to_html(), unsafe_allow_html=True)

if add_radio == "Reconciliation Manager":
    
    st.title("Reconciliation Manager")
    st.markdown("Load transactions to determine which have low reconciliation probability at close. Unreconciled items will be further analyzed with a secondary ML model to determine if there is a corresponding match elsewhere in systems.")
    st.markdown("The transactions with high reconiliation probability will continue through existing reconciliation processes. This limits unnecessary compute spend on easily reconciled resources.")
    initial_option = st.selectbox('Select Option', ['Select an option','Review Variables and Performance', 'Run Reconciliation Management'])
    if initial_option ==  'Run Reconciliation Management':
        file_upload()
    elif initial_option == 'Review Variables and Performance':
        display_model_info()

elif add_radio == "Budgeting and Forecasting":
    st.header('Budgeting and Forecasting')
    initial_selection = st.selectbox('Select Option', ['Select an option to begin','Review Previous Actuals', 'Submit Budgets for Review'])
    if initial_selection == 'Review Previous Actuals':
        st.subheader('Department Selection')
        department_selected = st.selectbox('Please select a department that you would like to review actuals and 2025 forecasts for.', department_list)
        st.markdown("This model uses a Seasonal Autoregressive Integrated Moving Average to more accurately determine future spend for 2024 and 2025. While 2024 is considered highly accurate, these models in their current state of this demo may show some variance in 2025.")
        go = st.button('Go')

        if go:
            plot_budget_projection(department_selected)
    elif initial_selection == 'Submit Budgets for Review':
        st.write('Under Construction')
