#%%
"""
Reference:
[1] Reimagining Synthetic Tabular Data Generation through Data-Centric AI: A Comprehensive Benchmark
- https://github.com/HLasse/data-centric-synthetic-data
"""
#%%
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
#%%
def MLu_cls(train_dataset, test_dataset, syndata):
    continuous = train_dataset.continuous_features
    target = train_dataset.ClfTarget
    
    train_ = train_dataset.raw_data.copy()
    test_ = test_dataset.raw_data.copy()
    syndata_ = syndata.copy()
    
    mean = train_[continuous].mean()
    std = train_[continuous].std() + 1e-6
    train_[continuous] -= mean
    train_[continuous] /= std
    test_[continuous] -= mean
    test_[continuous] /= std
    syndata_[continuous] -= mean
    syndata_[continuous] /= std
    
    covariates = [x for x in train_.columns if x not in [target]]

    """Baseline"""
    performance = []
    print(f"\n(Baseline) Classification: Accuracy...")
    for name, clf in [
        ('logit', LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000)),
        ('GaussNB', GaussianNB()),
        ('KNN', KNeighborsClassifier(n_jobs=-1)),
        ('SVM', SVC(random_state=42)),
        ('RF', RandomForestClassifier(random_state=42)),
    ]:
        clf.fit(train_[covariates], train_[target])
        pred = clf.predict(test_[covariates])
        acc = accuracy_score(test_[target], pred)
        if name == "RF":
            feature = [(x, y) for x, y in zip(covariates, clf.feature_importances_)]
        print(f"[{name}] ACC: {acc:.3f}")
        performance.append((name, acc))

    base_performance = performance
    base_cls = np.mean([x[1] for x in performance])
    base_feature = feature
    
    """Synthetic"""
    if syndata_[target].sum() == 0:
        return (
            base_cls, 0., 0., 0.
        )
    else:
        performance = []
        print(f"\n(Synthetic) Classification: Accuracy...")
        for name, clf in [
            ('logit', LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000)),
            ('GaussNB', GaussianNB()),
            ('KNN', KNeighborsClassifier(n_jobs=-1)),
            ('SVM', SVC(random_state=42)),
            ('RF', RandomForestClassifier(random_state=42)),
        ]:
            clf.fit(syndata_[covariates], syndata_[target])
            pred = clf.predict(test_[covariates])
            acc = accuracy_score(test_[target], pred)
            if name == "RF":
                feature = [(x, y) for x, y in zip(covariates, clf.feature_importances_)]
            print(f"[{name}] ACC: {acc:.3f}")
            performance.append((name, acc))
                
        syn_cls = np.mean([x[1] for x in performance])
        model_selection = spearmanr(
            np.array([x[1] for x in base_performance]),
            np.array([x[1] for x in performance])).statistic
        feature_selection = spearmanr(
            np.array([x[1] for x in base_feature]),
            np.array([x[1] for x in feature])).statistic
        
        return (
            base_cls, syn_cls, model_selection, feature_selection
        )
#%%