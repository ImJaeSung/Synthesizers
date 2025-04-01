#%%
import pandas as pd
from scipy.io.arff import loadarff 
#%%
def load_raw_data(dataset):
    if dataset == "banknote":
        data = pd.read_csv('./data/banknote.txt', header=None)
        data.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
        assert data.isna().sum().sum() == 0
        
        continuous_features = ["variance", "skewness", "curtosis", "entropy"]
        categorical_features = ['class']
        integer_features = []
        ClfTarget = "class"
        
    elif dataset == "whitewine":
        data = pd.read_csv('./data/whitewine.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = ["quality"]
        integer_features = []
        ClfTarget = "quality"
    
    elif dataset == "breast":
        data = pd.read_csv('./data/breast.csv')
        data = data.drop(columns=['id']) # drop ID number
        assert data.isna().sum().sum() == 0
        
        continuous_features = [x for x in data.columns if x != "diagnosis"]
        categorical_features = ["diagnosis"]
        integer_features = []
        ClfTarget = "diagnosis"
        
    elif dataset == "bankruptcy":
        data = pd.read_csv('./data/bankruptcy.csv')
        data.columns = [x.strip() for x in data.columns]
        assert data.isna().sum().sum() == 0
        
        continuous_features = [x for x in data.columns if x != "Bankrupt?"]
        categorical_features = ["Bankrupt?"]
        integer_features = [
            "Research and development expense rate",
            "Total Asset Growth Rate",
            "Inventory Turnover Rate (times)",
            "Quick Asset Turnover Rate",
            "Cash Turnover Rate",
            "Liability-Assets Flag",
            "Net Income Flag"
        ]
        ClfTarget = "Bankrupt?"
     
    elif dataset == "default":
        data = pd.read_excel('./data/default.xls', header=1)
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            'LIMIT_BAL',  
            'AGE', 
            'BILL_AMT1', 
            'BILL_AMT2',
            'BILL_AMT3',
            'BILL_AMT4', 
            'BILL_AMT5', 
            'BILL_AMT6', 
            'PAY_AMT1',
            'PAY_AMT2', 
            'PAY_AMT3', 
            'PAY_AMT4', 
            'PAY_AMT5', 
            'PAY_AMT6',
        ]
        categorical_features = [
            'SEX', 
            'EDUCATION', 
            'MARRIAGE', 
            'PAY_0',
            'PAY_2', 
            'PAY_3', 
            'PAY_4',
            'PAY_5', 
            'PAY_6', 
            'default payment next month'
        ]
        integer_features = [
            'LIMIT_BAL',  
            'AGE', 
        ]
        ClfTarget = "default payment next month"
    
    elif dataset == "BAF":
        # https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data
        data = pd.read_csv('./data/BAF.csv')
        
        ### remove missing values
        data = data.loc[data["prev_address_months_count"] != -1]
        data = data.loc[data["current_address_months_count"] != -1]
        data = data.loc[data["intended_balcon_amount"] >= 0]
        data = data.loc[data["bank_months_count"] != -1]
        data = data.loc[data["session_length_in_minutes"] != -1]
        data = data.loc[data["device_distinct_emails_8w"] != -1]
        data = data.reset_index(drop=True)
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            'income', 
            'name_email_similarity',
            'prev_address_months_count', 
            'current_address_months_count',
            'days_since_request', 
            'intended_balcon_amount',
            'zip_count_4w', 
            'velocity_6h', 
            'velocity_24h',
            'velocity_4w', 
            'bank_branch_count_8w',
            'date_of_birth_distinct_emails_4w', 
            'credit_risk_score', 
            'bank_months_count',
            'proposed_credit_limit', 
            'session_length_in_minutes', 
        ]
        categorical_features = [
            'customer_age', 
            'payment_type', 
            'employment_status',
            'email_is_free', 
            'housing_status',
            'phone_home_valid', 
            'phone_mobile_valid', 
            'has_other_cards', 
            'foreign_request', 
            'source',
            'device_os', 
            'keep_alive_session',
            'device_distinct_emails_8w', 
            'month',
            'fraud_bool', 
        ]
        integer_features = [
            'prev_address_months_count', 
            'current_address_months_count',
            'zip_count_4w', 
            'bank_branch_count_8w',
            'date_of_birth_distinct_emails_4w', 
            'credit_risk_score', 
            'bank_months_count',
        ]
        ClfTarget = "fraud_bool"

        
    return data, continuous_features, categorical_features, integer_features, ClfTarget