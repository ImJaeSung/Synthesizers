#%%
import pandas as pd
from scipy.io.arff import loadarff 
#%%
def load_raw_data(config):
    if config["dataset"] == "banknote":
        data = pd.read_csv('./data/banknote.txt', header=None)
        data.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
        assert data.isna().sum().sum() == 0
        
        continuous_features = ["variance", "skewness", "curtosis", "entropy"]
        categorical_features = ['class']
        integer_features = []
        ClfTarget = "class"
        
    elif config["dataset"] == "whitewine":
        data = pd.read_csv('./data/whitewine.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = ["quality"]
        integer_features = []
        ClfTarget = "quality"
    
    elif config["dataset"] == "breast":
        data = pd.read_csv('./data/breast.csv')
        data = data.drop(columns=['id']) # drop ID number
        assert data.isna().sum().sum() == 0
        
        continuous_features = [x for x in data.columns if x != "diagnosis"]
        categorical_features = ["diagnosis"]
        integer_features = []
        ClfTarget = "diagnosis"
        
    elif config["dataset"] == "bankruptcy":
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
        
    elif config["dataset"] == "musk":
        data = pd.read_csv('./data/musk.data', header=None)
        assert data.isna().sum().sum() == 0
            
        column = [i for i in range(1, 167)]
        columns = [
            'molecule_name', 
            'conformation_name'
        ] + [
            f"f{x}" for x in column
        ] + [
            'class'
        ]
        data.columns = columns
        columns.remove('molecule_name') 
        columns.remove('conformation_name')
        columns.remove('class') 
        continuous_features = columns
        categorical_features = [
            'molecule_name', 
            'conformation_name',
            'class', 
        ]
        integer_features = continuous_features
        ClfTarget = 'class'
    
    elif config["dataset"] == "madelon":
        data, _ = loadarff('./data/madelon.arff') # output : data, meta
        data = pd.DataFrame(data)
        assert data.isna().sum().sum() == 0

        for column in data.select_dtypes([object]).columns:
            data[column] = data[column].str.decode('utf-8') # object decoding
        continuous = [i for i in range(1, 501)]
        continuous_features = [f"V{x}" for x in continuous]
        categorical_features = ["Class"]
        integer_features = continuous_features
        ClfTarget = 'Class'
        
    elif config["dataset"] == "abalone":
        data = pd.read_csv('./data/abalone.data', header=None)
        columns = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
            "Rings",
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("Sex")
        columns.remove("Rings")
        continuous_features = columns
        categorical_features = [
            "Sex",
            "Rings"
        ]
        integer_features = []
        ClfTarget = "Rings"
    
    elif config["dataset"] == "anuran":
        data = pd.read_csv('./data/anuran.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [x for x in data.columns if x.startswith("MFCCs_")]
        categorical_features = [
            'Family',
            'Genus',
            'Species'
        ]
        integer_features = []
        ClfTarget = "Species"
        
    elif config["dataset"] == "shoppers":
        data = pd.read_csv('./data/shoppers.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            'Administrative', 
            'Administrative_Duration', 
            'Informational',
            'Informational_Duration', 
            'ProductRelated', 
            'ProductRelated_Duration',
            'BounceRates', 
            'ExitRates', 
            'PageValues', 
        ]
        categorical_features = [
            'SpecialDay', 
            'Month',
            'OperatingSystems', 
            'Browser', 
            'Region', 
            'TrafficType', 
            'VisitorType',
            'Weekend', 
            'Revenue'
        ]
        integer_features = [
            'Administrative', 
            'Informational',
            'ProductRelated', 
        ]
        ClfTarget = "Revenue"
        
    elif config["dataset"] == "default":
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
        
    elif config["dataset"] == "magic":
        data = pd.read_csv('./data/magic.data', header=None)
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            'fLength',
            'fWidth',
            'fSize',
            'fConc',
            'fConc1',
            'fAsym',
            'fM3Long',
            'fM3Trans',
            'fAlpha',
            'fDist',
        ]
        categorical_features = [
            'class'
        ]
        integer_features = [
        ]
        ClfTarget = "class"
        
        data.columns = continuous_features + categorical_features
        
    elif config["dataset"] == "madelon":
        data, _ = loadarff('./data/madelon.arff') # output : data, meta
        data = pd.DataFrame(data)
        assert data.isna().sum().sum() == 0

        for column in data.select_dtypes([object]).columns:
            data[column] = data[column].str.decode('utf-8') # object decoding
        continuous = [i for i in range(1, 501)]
        continuous_features = [f"V{x}" for x in continuous]
        categorical_features = ["Class"]
        integer_features = continuous_features
        ClfTarget = 'Class'
        
    return data, continuous_features, categorical_features, integer_features, ClfTarget
#%%