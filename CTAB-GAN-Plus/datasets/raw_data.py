#%%
import pandas as pd
from scipy.io.arff import loadarff 
#%%
def load_raw_data(dataset):
    if dataset == "banknote":
        data = pd.read_csv('../data/banknote.txt', header=None)
        data.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
        assert data.isna().sum().sum() == 0
        
        continuous_features = ["variance", "skewness", "curtosis", "entropy"]
        categorical_features = ['class']
        integer_features = []
        ClfTarget = "class"
        
    elif dataset == "whitewine":
        data = pd.read_csv('../data/whitewine.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = ["quality"]
        integer_features = []
        ClfTarget = "quality"
    
    elif dataset == "breast":
        data = pd.read_csv('../data/breast.csv')
        data = data.drop(columns=['id']) # drop ID number
        assert data.isna().sum().sum() == 0
        
        continuous_features = [x for x in data.columns if x != "diagnosis"]
        categorical_features = ["diagnosis"]
        integer_features = []
        ClfTarget = "diagnosis"
        
    elif dataset == "bankruptcy":
        data = pd.read_csv('../data/bankruptcy.csv')
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
        
    elif dataset == "musk":
        data = pd.read_csv('../data/musk.data', header=None)
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
        columns.remove('class') 
        columns.remove('molecule_name') 
        columns.remove('conformation_name')
        continuous_features = columns
        categorical_features = [
            'class', 
            'molecule_name', 
            'conformation_name'
        ]
        integer_features = continuous_features
        ClfTarget = 'class'
    
    elif dataset == "madelon":
        data, _ = loadarff('../data/madelon.arff') # output : data, meta
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