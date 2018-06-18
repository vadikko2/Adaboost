
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer


# In[57]:


def fit_preproc(x):
    colm_val_dict = {}
    for i in range(0, x.shape[1]):
        tmp = x[:, i]
        if type(tmp[0]) is str:
            count = 0
            value_dict = {}
            #составляем словарь
            for val in tmp:
                if not (val in value_dict.keys()):
                    if val == '?':
                        value_dict[val] = np.nan
                    else:
                        value_dict[val] = count
                        count+=1
            colm_val_dict[str(i)] = value_dict
    return colm_val_dict


# In[3]:


def transform(x, tr_dict):
    for i in range(0,x.shape[1]):
        if str(i) in tr_dict:
            tmp = x[:, i]
            for j in range(len(tmp)):
                if tmp[j] in tr_dict[str(i)]:
                    x[j, i] = tr_dict[str(i)][tmp[j]]
                else:
                    x[j, i] = len(tr_dict[str(i)])/2
    return x


# In[4]:


def sep_two_classes(Y, num_class):
    Y = Y.copy()
    for i in range(Y.shape[0]):
        if Y[i] == num_class:
            Y[i] = 1
        else:
            Y[i] = -1
    return Y


# In[58]:


def get_dataset():
    train = np.asarray(pd.read_csv('train2.csv'))
    test = np.asarray(pd.read_csv('test2.csv'))
    
    x_train , y_train = train[:, :-1], train[:,-1]
    x_test , y_test = test[:, :-1], test[:,-1]
    
    prepr_dict = fit_preproc(x_train)
    print('Dictionary:\n', prepr_dict)
    x_train = transform(x_train, prepr_dict)
    x_test = transform(x_test, prepr_dict)
    
    imp_train = Imputer(missing_values=np.nan)
    imp = imp_train.fit(x_train)
    x_train = imp.transform(x_train)
    x_test = imp.transform(x_test)

    y_train = sep_two_classes(y_train, '>50K')
    y_test = sep_two_classes(y_test, '>50K')

    pd.set_option('display.height',1000)
    pd.set_option('display.max_rows',10)
    pd.set_option('display.max_columns',500)
    pd.set_option('display.width',1000)

    df_train = pd.DataFrame([np.append(x_train[i], y_train[i]) for i in range(x_train.shape[0])], columns = [str(i) for i in range(x_train.shape[1]+1)])
    print('\nTrain data:\n', df_train)

    df_test = pd.DataFrame([np.append(x_test[i], y_test[i]) for i in range(x_test.shape[0])], columns = [str(i) for i in range(x_test.shape[1]+1)])
    print('\nTest data:\n', df_test)
    return x_train, y_train, x_test, y_test

