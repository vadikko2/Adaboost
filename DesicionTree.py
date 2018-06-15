
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


# In[3]:


def fit_preproc(x):
    colm_val_dict = {}
    for i in range(0,x.shape[1]):
        tmp = x[:, i]
        if type(tmp[0]) is str:
            count = 0
            value_dict = {}
            #составляем словарь
            for val in tmp:
                if not (val in value_dict.keys()):
                    if val == ' ?':
                        value_dict[val] = np.nan
                    else:
                        value_dict[val] = count
                        count+=1
            colm_val_dict[str(i)] = value_dict
    return colm_val_dict


# In[4]:


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


# In[54]:


def get_dataset(filename):
    train = np.asarray(pd.read_csv(filename))
    X , Y = train[:, :-1], train[:,-1]
    p=np.random.permutation(X.shape[0])
    x_train = X[p[0:int(len(X)*0.7)], :]
    y_train = Y[p[0:int(len(Y)*0.7)]]
    x_test = X[p[int(len(X)*0.7):], :]
    y_test = Y[p[int(len(Y)*0.7):]]

    prepr_dict = fit_preproc(x_train)
    print('Dictionary:\n', prepr_dict)
    x_train = transform(x_train, prepr_dict)
    x_test = transform(x_test, prepr_dict)

    imp_train = Imputer(missing_values=np.nan)
    imp = imp_train.fit(x_train)
    x_train = imp.transform(x_train)
    x_test = imp.transform(x_test)
    
    y_train = sep_two_classes(y_train, 1)
    y_test = sep_two_classes(y_test, 1)

    pd.set_option('display.height',1000)
    pd.set_option('display.max_rows',10)
    pd.set_option('display.max_columns',500)
    pd.set_option('display.width',1000)

    df_train = pd.DataFrame([np.append(x_train[i], y_train[i]) for i in range(x_train.shape[0])], columns = [str(i) for i in range(x_train.shape[1]+1)])
    print('\nTrain data:\n', df_train)

    df_test = pd.DataFrame([np.append(x_test[i], y_test[i]) for i in range(x_test.shape[0])], columns = [str(i) for i in range(x_test.shape[1]+1)])
    print('\nTest data:\n', df_test)

    x_train = (x_train - np.mean(x_train, axis = 0)) / np.var(x_train, axis = 0)
    x_test = (x_test - np.mean(x_test, axis = 0)) / np.var(x_test, axis = 0)
    return x_train, y_train, x_test, y_test


# In[8]:


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# In[9]:


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# In[22]:


def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}


# In[23]:


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# In[24]:


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)


# In[25]:


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# In[26]:


# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))


# In[37]:


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# In[42]:


def accuracy(y_true, y_pred):
    counter = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            counter +=1
    return counter/len(y_true)


# In[60]:


def get_tree(x_train, y_train, max_depth, min_size):
    train_dataset = [np.append(x_train[i], y_train[i]) for i in range(x_train.shape[0])]
    tree = build_tree(train_dataset, max_depth, min_size)
    return tree


# In[63]:


def test():
    x_train, y_train, x_test, y_test = get_dataset('train2.csv')
    test_dataset = [np.append(x_train[i], y_train[i]) for i in range(x_train.shape[0])][:200]
    tree = get_tree(x_train[:200], y_train[:200], 10, 10)
    print('\nTree\n')
    print_tree(tree)
    predictions = []
    for row in test_dataset:
        predictions.append(predict(tree, row))
    print('\naccuracy:', accuracy(y_test[:200], predictions))

