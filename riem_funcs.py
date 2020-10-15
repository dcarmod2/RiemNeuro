from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import scipy
import os
import random
import pyriemann
import re
import itertools

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.distance import distance_riemann

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier

def get_label_8(str_inp,loc):
    if loc == 'chm':
        to_add = 0
    elif loc == 'san':
        to_add = 1
    else:
        raise ValueError("Invalid location")
    
    if str_inp == 'ctr':
        return 0 + to_add
    if str_inp == 'hl':
        return 2 + to_add
    if str_inp == 'tin_hl':
        return 4 + to_add
    if str_inp == 'tin':
        return 6 + to_add

def data_selector(sample,valid_runs_dict_uiuc,valid_runs_dict_whasc):
    if sample['Location'] == 'chm':
        cond1 = int(sample['Run'].lstrip('run')) in valid_runs_dict_uiuc.get(sample['ID'],[])
    elif sample['Location'] == 'san':
        cond1 = int(sample['Run'].lstrip('run')) in valid_runs_dict_whasc.get(sample['ID'],[])
    else:
        raise Exception("Invalid Location")
    cond2 = int(sample['Run'].lstrip('run')) != 3
    return cond1 and cond2

def data_selector_music(sample,valid_runs_dict_uiuc,valid_runs_dict_whasc):
    cond1 = int(re.sub(r'sub_','',sample['Name'])) in metadata.index
    cond2 = int(sample['Run'].lstrip('run')) == 3
    return cond1 and cond2

def get_TFI(sample):
    if sample["Location"] == "chm":
        return metadata_uiuc.loc[int(sample['ID']),'TFI_A']
    if sample["Location"] =="san":
        return metadata_whasc.loc[int(sample['ID']),'TFI_A']

class Correlations(BaseEstimator, TransformerMixin):
    def __init__(self):
        return
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return np.array([np.corrcoef(x) for x in X])
    
    def fit_transform(self,X,y=None):
        return np.array([np.corrcoef(x) for x in X])

class PCA_coord_change(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pca_instance = PCA()
        
    def fit(self,X,y=None):
        
        self.pca_instance.fit(X,y)
        return self
    
    def transform(self,X):
        shape = X.shape
        coords = self.pca_instance.components_
        coord_shape = coords.shape
        if coord_shape[1] < shape[0]:
            coords_padded = np.hstack([coords, np.zeros((coord_shape[0],-coord_shape[1]+shape[0]))])
            X_padded = X
        else:
            coords_padded = coords
            X_padded = np.vstack([X,np.zeros((coord_shape[1]-shape[0],shape[1]))])
        return np.dot(coords_padded,X_padded)
       
    
    def fit_transform(self,X,y=None,sample_weight=None):
        
        self.pca_instance.fit(X,y)
        shape = X.shape
        coords = self.pca_instance.components_
        coord_shape = coords.shape
        if coord_shape[1] < shape[0]:
            coords_padded = np.hstack([coords, np.zeros((coord_shape[0],-coord_shape[1]+shape[0]))])
            X_padded = X
        else:
            coords_padded = coords
            X_padded = np.vstack([X,np.zeros((coord_shape[1]-shape[0],shape[1]))])
        return np.dot(coords_padded,X_padded)
        
    
class to_upper_tri(BaseEstimator, TransformerMixin):
    def __init__(self,k):
        self.k = k
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        shape = X[0].shape
        inds = np.triu_indices(shape[0],self.k)
        return np.array([x[inds].flatten() for x in X])
    
    def fit_transform(self,X,y=None,sample_weight=None):
        shape = X[0].shape
        inds = np.triu_indices(shape[0],self.k)
        return np.array([x[inds].flatten() for x in X])
    
class to_diag(BaseEstimator, TransformerMixin):
    def __init__(self,rows):
        self.rows=rows
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return np.array([np.array([x[i,i] for i in range(0,self.rows)]).flatten() for x in X])
    
    def fit_transform(self,X,y=None,sample_weight=None):
        shape = X[0].shape
        
        return np.array([np.array([x[i,i] for i in range(0,self.rows)]).flatten() for x in X])

    
class to_symm_mat(BaseEstimator, TransformerMixin):
    def __init__(self,k,rows):
        self.k = k
        self.rows=rows
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        inds = np.triu_indices(self.rows,self.k)
        to_ret = []
        for x in X:
            zeroed = np.zeros(shape=(self.rows,self.rows))
            zeroed[inds] = x
            to_ret.append((zeroed+zeroed.T)/(np.eye(self.rows)+np.ones((self.rows,self.rows))))
        return np.array(to_ret)
    
    def fit_transform(self,X,y=None,sample_weight=None):
        inds = np.triu_indices(self.rows,self.k)
        to_ret = []
        for x in X:
            zeroed = np.zeros(shape=(self.rows,self.rows))
            zeroed[inds] = x
            to_ret.append((zeroed+zeroed.T)/(np.eye(self.rows)+np.ones((self.rows,self.rows))))
        return np.array(to_ret)


def get_sig_pairs(coeffarr,nullcoeffs,num_pairs,p,rois,maxmin="max"):
    def ind_to_pair(ind):
        x,y = np.triu_indices(33)
        return (x[ind],y[ind])
    if maxmin == "max":
        percentile = 100-p
    elif maxmin == "min":
        percentile = p
    thresholds = np.percentile(nullcoeffs,percentile,axis=0)
    if maxmin == "max":
        boolarr =  coeffarr > thresholds.reshape(num_pairs,1)
    elif maxmin == "min":
        boolarr = coeffarr < thresholds.reshape(num_pairs,1)
    indarr = [[(rois[ind_to_pair(i)[0]],rois[ind_to_pair(i)[1]]) 
     for i in range(0,len(boolarr[0])) if boolarr[j][i]] for j in range(0,len(boolarr))]
    return indarr,boolarr


def permutation_bootstrap(samples,labels,n_states,rois,p=5):
    '''
    Perform permutation bootstrap to find discriminative connections.
    
        Parameters:
            samples (ndarray shape (n_samples, n_channels, n_vars)): The input dataset
            labels (ndarray shape (n_samples)): The input labels
            n_states (int): Number of distinct classes
            p (int): Percentile of significance (default 5)
            
        Returns:
            discrim_conn_max (list of tuples): Significant functional positive connections
            discrim_conn_min (list of tuples): Significant functional negative connections
    '''
    # First, make null model
    # Use integer labeling so we can be sure that the one vs one classifiers are 
    # in the correct orders
    X = samples
    y = labels
    
    # Randomly permute labels (only labels, not training input)
    NUM_BOOTSTRAP = 10
    covest = Covariances()  
    ts = TangentSpace()
    sym = to_symm_mat(0,X.shape[1])
    diag = to_upper_tri(1)
    svc = SVC(kernel='linear')
    clf_riem = make_pipeline(covest,ts,sym,diag,svc)
    maxcoeffs = []   
    mincoeffs = []
    nullcoeffs = []
    nullcos = []
    num_pairs = int(scipy.special.binom(n_states,2))
    for i in range(0,100):
        y_permuted = np.random.permutation(y)
        coeffArr = []
        rs = ShuffleSplit(n_splits=NUM_BOOTSTRAP, test_size=.3)
        for train,test in rs.split(X):
            X_train, X_test, y_train, y_test = X[train],X[test],y_permuted[train],y_permuted[test]
            clf_riem.fit(X_train,y_train)
            coeffArr.append(clf_riem[4].coef_/np.std(clf_riem[4].coef_,axis=-1).reshape(num_pairs,1))

        meancoeff = sum(coeffArr)/len(coeffArr)
        classcos = []
        for z in range(0,len(coeffArr[0])):
            class_z_coeffs = [x[z] for x in coeffArr]
            cos_sim = cosine_similarity(class_z_coeffs)
            upperTri = cos_sim[np.triu_indices(cos_sim.shape[0],1)]
            cos_max = np.max(upperTri.flatten())
            classcos.append(cos_max)

        nullcos.append(classcos)
        nullcoeffs.append(meancoeff)
        maxcoeff = np.max(meancoeff,axis=-1)
        mincoeff = np.min(meancoeff,axis=-1)
        maxcoeffs.append(maxcoeff)
        mincoeffs.append(mincoeff)

    coeffArr = []
    rs = ShuffleSplit(n_splits=NUM_BOOTSTRAP, test_size=.3)
    for train,test in rs.split(X):
        X_train, X_test, y_train, y_test = X[train],X[test],y[train],y[test]
        clf_riem.fit(X_train,y_train)
        coeffArr.append(clf_riem[4].coef_/np.std(clf_riem[4].coef_,axis=-1).reshape(num_pairs,1))
    meancoeff = sum(coeffArr)/len(coeffArr)

    sig_pairs_max, boolarr_max = get_sig_pairs(meancoeff,maxcoeffs,num_pairs,p,rois,"max")
    sig_pairs_min, boolarr_min = get_sig_pairs(meancoeff,mincoeffs,num_pairs,p,rois,"min")
    
    combs = list(itertools.combinations(range(0,n_states),2))
    #discrim_conn_max = [[combs[z] for z,flag in enumerate(one_vs_one_conns) if flag] for one_vs_one_conns in boolarr_max]
    #discrim_conn_min = [[combs[z] for z,flag in enumerate(one_vs_one_conns) if flag] for one_vs_one_conns in boolarr_min]

    
    return sig_pairs_max, sig_pairs_min

def generate_test(n_samples = 200,n_vars=500,n_channels = 5,n_states=3,rois = [0,1,2,3,4], correlated_inds = [[0,1],[0,2],[0,3]]):
    '''
    Produces test data and tests whether projecting matrices into the tangent space finds the correct discriminative connection.
    
        Parameters:
            n_samples (int): Number of samples per state
            n_vars (int): Number of columns in each time series
            n_channels (int): Number of channels in time series
            n_states (int): Number of distinct classes
            correlated_inds (list of lists): The indices of the channels in each state which are almost perfectly correlated.
            
        Returns:
            discrim_conn_max (list of tuples): Significant functional positive connections
            discrim_conn_min (list of tuples): Significant functional negative connections
            samples (ndarray shape (n_samples, n_channels, n_vars)): The constructed dataset
            labels (ndarray shape (n_samples)): The constructed labels
    '''
    
    
    samples = []
    labels = []
    for i in range(0,n_samples):
        for j in range(0,n_states):
            corr_chans = correlated_inds[j]
            correlated = np.random.normal(size = n_vars,scale = 1)
            sample = []
            for k in range(0,n_channels):
                if k in corr_chans:
                    noise = np.random.normal(size = n_vars, scale = 0.1)
                    sample.append(correlated+noise)
                else:
                    sample.append(np.random.normal(size = n_vars,scale = 1))
            samples.append(sample)
            labels.append(j)
        
    
    samples = np.array(samples)
    labels = np.array(labels)
    return samples,labels,permutation_bootstrap(samples,labels,n_states,rois=rois,p=5)

def train_classifiers(data_files,valid_runs_dict_uiuc,valid_runs_dict_whasc):
    '''
    Produces test data and tests whether projecting matrices into the tangent space finds the correct discriminative connection.
    
        Parameters:
            data_files (list of pairs (filename,data)): the input data
            valid_runs_dict_uiuc (dictionary): dictionary containing valid runs for each patient
            valid_runs_dict_whasc (dictionary): dictionary containing valid runs for each patient
            
        Returns:
            accDict (dictionary): mean accuracy on each file's data
            simDict (dictionary): mean cosine similarity of classifier coefficients for each file
            matDict (dictionary): mean confusion matrix for each file
            corrDict (dictionary): before and after projection correlations
            spearDict (dictionary): before and after projection spearman correlations
    '''
    accDict = {}
    simDict = {}
    matDict = {}
    corrDict = {}
    spearDict = {}
    
    simArr = []
    for fname,data in data_files:
        # get time series data to make covariance matrices
        X = np.array([sample['TimeSeries'] for sample in data['samples']])# if data_selector(sample)])        
        y = np.array([get_label_8(sample['Group'],sample['Location']) for sample in data['samples']])# if data_selector(sample)])

        # gsr seems to produce a rank deficient covariance matrix, so oas regularization is necessary
        covest = Covariances()  
        ts = TangentSpace()
        #sym = to_symm_mat(0,33)
        #diag = to_upper_tri(1)
        svc = SVC(kernel='linear')
        clf_riem = make_pipeline(covest,ts,svc)

        rf = RandomForestClassifier(200)
        clf_rf = make_pipeline(covest,ts,rf)


        covest2 = Correlations()

        svc2 = SVC(kernel='linear')
        get_tri_inds = to_upper_tri(0)
        clf_cov = make_pipeline(covest2,get_tri_inds,svc2)


        #Check clustering
        #to_TS = make_pipeline(covest,ts)
        #X_in_TS = to_TS.transform(X)
        #kmeans = KMeans(n_clusters=4,random_state=0).fit(X_in_TS)

        # Monte Carlo, in theory should run this len(y)^2 times, but I need to save my poor computer's memory.
        accRiemList = []
        accCovList = []
        accRfList = []
        coeffArr = []
        matRiemList = []
        corrArrBefore = []
        corrArrAfter = []
        spearArrBefore = []
        spearArrAfter = []

        rs = StratifiedShuffleSplit(n_splits=100, test_size=.3)
        for i,(train_inds,test_inds) in enumerate(rs.split(X,y)):

            X_train, X_test, y_train, y_test = X[train_inds],X[test_inds],y[train_inds],y[test_inds]
            X_train_cov, X_test_cov, y_train_cov, y_test_cov = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()

            clf_riem.fit(X_train,y_train)
            clf_rf.fit(X_train,y_train)
            clf_cov.fit(X_train_cov,y_train_cov)


            #get riemann svm coefficients
            coeffArr.append(clf_riem[2].coef_)

            #compare correlation
            corr_coeffs_before = np.corrcoef(np.vstack([x[np.triu_indices(33)].flatten() for x in X_train]),rowvar=False)
            corrArrBefore.append(np.linalg.norm(corr_coeffs_before))
            #spearman correlation
            spearman_coeffs_before,_ = scipy.stats.spearmanr(np.vstack([x[np.triu_indices(33)].flatten() for x in X_train]),axis=0)
            spearArrBefore.append(np.linalg.norm(spearman_coeffs_before))

            ref = ts.reference_
            covs = covest.transform(X_train)
            mapped = ts.transform(covs)
            corr_coeffs_after = np.corrcoef(mapped,rowvar=False)
            spearman_coeffs_after = scipy.stats.spearmanr(mapped,axis=0)
            corrArrAfter.append(np.linalg.norm(corr_coeffs_after))
            spearArrAfter.append(np.linalg.norm(spearman_coeffs_after))

            y_pred = clf_riem.predict(X_test)
            y_pred_cov = clf_cov.predict(X_test_cov)
            y_pred_rf = clf_rf.predict(X_test)

            # save accuracy
            accRiemList.append(accuracy_score(y_pred,y_test))
            accCovList.append(accuracy_score(y_pred_cov,y_test_cov))
            accRfList.append(accuracy_score(y_pred_rf,y_test))

            # confusion matrix
            mat = confusion_matrix(y_test,y_pred,normalize='true',labels=[0,1,2,3,4,5,6,7])
            matRiemList.append(mat)

        for z in range(0,len(coeffArr[0])):
            class_z_coeffs = [x[z] for x in coeffArr]
            cos_sim = cosine_similarity(class_z_coeffs)
            upperTri = cos_sim[np.triu_indices(cos_sim.shape[0],1)]
            cos_avg = np.mean(upperTri.flatten())
            simArr.append(cos_avg)

        avgMatRiem = sum(matRiemList)/len(matRiemList)
        simDict.update({fname: simArr}) 
        matDict.update({fname: avgMatRiem})
        riemAcc = np.mean(accRiemList)
        covAcc = np.mean(accCovList)
        rfAcc = np.mean(accRfList)

        accDict.update({'raw_data':{'riem':riemAcc, 'rf':rfAcc, 'cov':covAcc}})
        corrDict.update({'raw_data':{'before':np.mean(corrArrBefore),'after':np.mean(corrArrAfter)}}) 
        spearDict.update({'raw_data':{'before':np.mean(spearArrBefore),'after':np.mean(spearArrAfter)}}) 
        print("Mean Accuracy w/ Riemann on data " + fname + ": " + str(riemAcc))
        print("Mean Accuracy w/ Cov on data " + fname + ": " + str(covAcc))
        print("Mean Accuracy w/ RF on data " + fname + ": " + str(rfAcc))
        print("----------------")
    
    return accDict, corrDict, spearDict, matDict, simDict
