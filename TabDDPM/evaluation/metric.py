# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

from statsmodels.distributions.empirical_distribution import ECDF

from scipy.stats import wasserstein_distance
from scipy.stats import spearmanr
from scipy.spatial import distance_matrix
#%%
def MLutility(train_dataset, test_dataset, syndata):
    continuous = train_dataset.continuous_features
    discrete = train_dataset.categorical_features
    target = train_dataset.ClfTarget
    
    """Baseline"""
    train_ = train_dataset.raw_data.copy()
    mean = train_[continuous].mean()
    std = train_[continuous].std()
    train_[continuous] -= mean
    train_[continuous] /= std
    test = test_dataset.raw_data.copy()
    test[continuous] -= mean
    test[continuous] /= std
    covariates = [x for x in train_.columns if x not in [target]]

    performance = []
    print(f"Target: {target}")
    for name, clf in [
        ('logit', LogisticRegression(random_state=0, n_jobs=-1)),
        ('GaussNB', GaussianNB()),
        ('KNN', KNeighborsClassifier(n_jobs=-1)),
        ('tree', DecisionTreeClassifier(random_state=0)),
        ('RF', RandomForestClassifier(random_state=0)),
    ]:
        clf.fit(train_[covariates], train_[target])
        pred = clf.predict(test[covariates])
        f1 = f1_score(test[target], pred, average='micro')
        # try:
        #     pred = clf.predict_proba(test[covariates])
        #     auroc = roc_auc_score(test[target].to_list(), pred, multi_class='ovo', average='macro')
        # except:
        #     pred = clf.predict(test[covariates])
        #     auroc = roc_auc_score(test[target].to_list(), pred, average='macro')
        if name == "RF":
            feature = [(x, y) for x, y in zip(covariates, clf.feature_importances_)]
        print(f"[{name}] F1: {f1:.3f}")
        performance.append((name, f1))

    base_performance = performance
    base_cls_performance = np.mean([x[1] for x in performance])
    base_feature = feature
    
    """Synthetic"""
    syndata_ = syndata.copy()
    mean = syndata_[continuous].mean()
    std = syndata_[continuous].std()
    syndata_[continuous] -= mean
    syndata_[continuous] /= std
    test = test_dataset.raw_data.copy()
    test[continuous] -= mean
    test[continuous] /= std
    covariates = [x for x in syndata_.columns if x not in [target]]

    performance = []
    print(f"Target: {target}")
    for name, clf in [
        ('logit', LogisticRegression(random_state=0, n_jobs=-1)),
        ('GaussNB', GaussianNB()),
        ('KNN', KNeighborsClassifier(n_jobs=-1)),
        ('tree', DecisionTreeClassifier(random_state=0)),
        ('RF', RandomForestClassifier(random_state=0)),
    ]:
        clf.fit(syndata_[covariates], syndata_[target])
        pred = clf.predict(test[covariates])
        f1 = f1_score(test[target], pred, average='micro')
        # try:
        #     pred = clf.predict_proba(test[covariates])
        #     auroc = roc_auc_score(test[target].to_list(), pred, multi_class='ovo', average='macro')
        # except:
        #     pred = clf.predict(test[covariates])
        #     auroc = roc_auc_score(test[target].to_list(), pred, average='macro')
        if name == "RF":
            feature = [(x, y) for x, y in zip(covariates, clf.feature_importances_)]
        print(f"[{name}] F1: {f1:.3f}")
        performance.append((name, f1))
            
    cls_performance = np.mean([x[1] for x in performance])
    
    model_selection = spearmanr(
        np.array([x[1] for x in base_performance]),
        np.array([x[1] for x in performance])).statistic

    feature_selection = spearmanr(
        np.array([x[1] for x in base_feature]),
        np.array([x[1] for x in feature])).statistic
    
    return (
        base_cls_performance, cls_performance, model_selection, feature_selection
    )
#%%
def statistical_similarity(train, syndata):
    train = train.to_numpy()
    syndata = syndata.to_numpy()
    
    Dn_list = []
    W1_list = []
    for j in range(train.shape[1]):
        xj = train[:, j]
        ecdf = ECDF(xj)
        ecdf_hat = ECDF(syndata[:, j])

        Dn = np.abs(ecdf(xj) - ecdf_hat(xj)).max()
        W1 = wasserstein_distance(xj, syndata[:, j])
        
        Dn_list.append(Dn)
        W1_list.append(W1)
    return Dn_list, W1_list
#%%
def DCR_metric(train, syndata, data_percent=15):
    
    """
    Reference:
    [1] https://github.com/Team-TUD/CTAB-GAN/blob/main/model/eval/evaluation.py
    
    Returns Distance to Closest Record
    
    Inputs:
    1) train -> real data
    2) synthetic -> corresponding synthetic data
    3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing Distance to Closest Record
    Outputs:
    1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
    along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets
    
    """
    
    # Sampling smaller sets of real and synthetic data to reduce the time complexity of the evaluation
    real_sampled = train.sample(n=int(len(train)*(.01*data_percent)), random_state=42).to_numpy()
    fake_sampled = syndata.sample(n=int(len(syndata)*(.01*data_percent)), random_state=42).to_numpy()

    # Computing pair-wise distances between real and synthetic 
    dist_rf = metrics.pairwise_distances(real_sampled, Y=fake_sampled, metric='minkowski', n_jobs=-1)
    # Computing pair-wise distances within real 
    dist_rr = metrics.pairwise_distances(real_sampled, Y=None, metric='minkowski', n_jobs=-1)
    # Computing pair-wise distances within synthetic
    dist_ff = metrics.pairwise_distances(fake_sampled, Y=None, metric='minkowski', n_jobs=-1) 
    
    # Removes distances of data points to themselves to avoid 0s within real and synthetic 
    rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0],dtype=bool)].reshape(dist_rr.shape[0],-1)
    rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0],dtype=bool)].reshape(dist_ff.shape[0],-1) 
    
    # Computing first and second smallest nearest neighbour distances between real and synthetic
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]       
    # Computing first and second smallest nearest neighbour distances within real
    smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
    smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]
    # Computing first and second smallest nearest neighbour distances within synthetic
    smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
    smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]
    
    # Computing 5th percentiles for DCR and NNDR between and within real and synthetic datasets
    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf,5)
    min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    fifth_perc_rr = np.percentile(min_dist_rr,5)
    min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    fifth_perc_ff = np.percentile(min_dist_ff,5)
    # nn_ratio_rf = np.array([i[0]/i[1] for i in smallest_two_rf])
    # nn_fifth_perc_rf = np.percentile(nn_ratio_rf,5)
    # nn_ratio_rr = np.array([i[0]/i[1] for i in smallest_two_rr])
    # nn_fifth_perc_rr = np.percentile(nn_ratio_rr,5)
    # nn_ratio_ff = np.array([i[0]/i[1] for i in smallest_two_ff])
    # nn_fifth_perc_ff = np.percentile(nn_ratio_ff,5)
    
    return [fifth_perc_rf,fifth_perc_rr,fifth_perc_ff]
    # return np.array([fifth_perc_rf,fifth_perc_rr,fifth_perc_ff,nn_fifth_perc_rf,nn_fifth_perc_rr,nn_fifth_perc_ff]).reshape(1,6) 
#%%
def regression_eval(train, test, target, mean, std):
    train[target] = train[target] * std[target] + mean[target]
    test[target] = test[target] * std[target] + mean[target]
    
    covariates = [x for x in train.columns if x not in [target]]
    
    result = []
    name = 'RF'
    regr = RandomForestRegressor(random_state=0, n_jobs=-1)
    regr.fit(train[covariates], train[target])
    pred = regr.predict(test[covariates])
    
    mare = (test[target] - pred).abs()
    mare /= test[target].abs() + 1e-6
    mare = mare.mean()
    
    result.append((name, mare))
    print("[{}] MAPE: {:.3f}".format(name, mare))
    return result
#%%
def classification_eval(train, test, target):
    covariates = [x for x in train.columns if not x.startswith(target)]
    train_target = train[target]
    test_target = test[target].to_numpy()
    
    result = []
    name = 'RF'
    clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    clf.fit(train[covariates], train_target)
    pred = clf.predict(test[covariates])
    
    f1 = f1_score(test_target, pred, average='micro')
    
    result.append((name, f1))
    print("[{}] F1: {:.3f}".format(name, f1))
    return result
#%%
def attribute_disclosure(K, compromised, syndata, attr_compromised, dataset):
    dist = distance_matrix(
        compromised[attr_compromised].to_numpy(),
        syndata[attr_compromised].to_numpy(),
        p=2)
    K_idx = dist.argsort(axis=1)[:, :K]
    
    def most_common(lst):
        return max(set(lst), key=lst.count)
    
    votes = []
    trues = []
    for i in tqdm(range(len(K_idx)), desc="Marjority vote..."):
        true = np.zeros((len(dataset.categorical_features), ))
        vote = np.zeros((len(dataset.categorical_features), ))
        for j in range(len(dataset.categorical_features)):
            true[j] = compromised.to_numpy()[i, len(dataset.continuous_features) + j]
            vote[j] = most_common(list(syndata.to_numpy()[K_idx[i], len(dataset.continuous_features) + j]))
        votes.append(vote)
        trues.append(true)
    votes = np.vstack(votes)
    trues = np.vstack(trues)
    
    acc = 0
    f1 = 0
    for j in range(trues.shape[1]):
        acc += (trues[:, j] == votes[:, j]).mean()
        f1 += f1_score(trues[:, j], votes[:, j], average="macro", zero_division=0)
    acc /= trues.shape[1]
    f1 /= trues.shape[1]

    return acc, f1
#%%
def marginal_plot(train, syndata, config):
    if not os.path.exists(f"./assets/figs/{config['dataset']}/seed{config['seed']}/"):
        os.makedirs(f"./assets/figs/{config['dataset']}/seed{config['seed']}/")
    
    figs = []
    for idx, feature in tqdm(enumerate(train.columns), desc="Plotting Histograms..."):
        fig = plt.figure(figsize=(7, 4))
        fig, ax = plt.subplots(1, 1)
        sns.histplot(
            data=syndata,
            x=syndata[feature],
            stat='density',
            label='synthetic',
            ax=ax,
            bins=int(np.sqrt(len(syndata)))) 
        sns.histplot(
            data=train,
            x=train[feature],
            stat='density',
            label='train',
            ax=ax,
            bins=int(np.sqrt(len(train)))) 
        ax.legend()
        ax.set_title(f'{feature}', fontsize=15)
        plt.tight_layout()
        plt.savefig(f"./assets/figs/{config['dataset']}/seed{config['seed']}/hist_{feature}.png")
        # plt.show()
        plt.close()
        figs.append(fig)
    return figs
#%%