# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, boxcar
from numpy import convolve
from sklearn.linear_model import LogisticRegression
from glob import glob
import os
from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler
from sklearn.qda import QDA
from sklearn.ensemble import RandomForestClassifier

 
#############function to read data###########

def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    
    # events file
    events_fname = fname.replace('_data','_events')
    
    # read event file
    labels= pd.read_csv(events_fname)
    
    # clean ids
    clean=data.drop(['id' ], axis=1)
    labels=labels.drop(['id' ], axis=1)
    
    return  clean,labels

def butterworth_filter(X,k,l):
    b,a = butter(3,k/500.0,btype='lowpass')
    X = lfilter(b,a,X)
    return X

def prepare_data_test(fname):
    """ read and prepare test data """
    # Read data
    data = pd.read_csv(fname)
    return data

scaler= StandardScaler()

# Use low pass filters to filter out unnecessary brain waves
def data_preprocess_train(X):
    X_prep_normal = scaler.fit_transform(X)
    X_prep_low = np.zeros((np.shape(X_prep_normal)[0],10))
    for i in range(10):
        X_prep_low[:,i] = butterworth_filter(X[:,0],2-(i*0.2),3)
        X_prep_low[:,i] = scaler.fit_transform(X_prep_low[:,i])
    X_prep_low_pow = X_prep_low ** 2
    X_prep = np.concatenate((X_prep_low,X_prep_normal,X_prep_low_pow),axis=1)

    return X_prep

def data_preprocess_test(X):
    X_prep_normal = scaler.fit_transform(X)
    X_prep_low = np.zeros((np.shape(X_prep_normal)[0],10))
    for i in range(10):
        X_prep_low[:,i] = butterworth_filter(X[:,0],2-(i*0.2),3)
        X_prep_low[:,i] = scaler.fit_transform(X_prep_low[:,i])
    X_prep_low_pow = X_prep_low ** 2
    X_prep = np.concatenate((X_prep_low,X_prep_normal,X_prep_low_pow),axis=1)
    
    return X_prep

# Subsample training
subsample  = 1000
subsample2 = 150

#######columns name for labels#############
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

#######number of subjects###############
subjects = range(1,13)
ids_tot = []
pred_tot = []

###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    y_raw= []
    raw = []
    ################ READ DATA ################################################
    fnames =  glob('input/train/subj%d_series*_data.csv' % (subject))
    for fname in fnames:
      data,labels=prepare_data_train(fname)
      raw.append(data)
      y_raw.append(labels)

    X = pd.concat(raw)
    y = pd.concat(y_raw)
    #transform in numpy array
    #transform train data in numpy array
    X_train =np.asarray(X.astype(float))
    y = np.asarray(y.astype(float))


    ################ Read test data #####################################
    #
    fnames =  glob('input/test/subj%d_series*_data.csv' % (subject))
    test = []
    idx=[]
    for fname in fnames:
      data=prepare_data_test(fname)
      test.append(data)
      idx.append(np.array(data['id']))
    X_test= pd.concat(test)
    ids=np.concatenate(idx)
    ids_tot.append(ids)
    X_test=X_test.drop(['id' ], axis=1)#remove id
    #transform test data in numpy array
    X_test =np.asarray(X_test.astype(float))


    ################ Train classifiers ########################################
    lda = LDA()
    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, criterion="entropy", random_state=1)
    lr2 = LogisticRegression()
    
    pred1 = np.empty((X_test.shape[0],6))
    pred2 = np.empty((X_test.shape[0],6))
    pred3 = np.empty((X_test.shape[0],6))

    pred = np.empty((X_test.shape[0],6))
    
    X_train=data_preprocess_train(X_train)
    X_test=data_preprocess_test(X_test)
    for i in range(6):
        y_train= y[:,i]
        print('Train subject %d, class %s' % (subject, cols[i]))
        
        # Fit models
        lda.fit(X_train,y_train)
        rf.fit(X_train, y_train)
        lr2.fit(X_train,y_train)
        
        # Grab predictions
        pred1[:,i] = lda.predict_proba(X_test)[:,1]
        pred2[:,i] = rf.predict_proba(X_test)[:,1]
        pred3[:,i] = lr2.predict_proba(X_test)[:,1]
        
        # Ensemble!
        pred[:,i]=(pred1[:,i] + pred2[:,i] + pred3[:,i])/3

    pred_tot.append(pred)

# submission file
#lda_file = 'lda.csv'
lda_file = 'lda_rf.csv'

# create pandas object for sbmission

lda = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))
# write file

lda.to_csv(lda_file,index_label='id',float_format='%.3f')

print('Done!')