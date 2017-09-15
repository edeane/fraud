import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import re
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import pickle

from model import Fraud

def predict_stuff(fraud_pkl):

    a_row = df.iloc[0,:]
    a_row_df = pd.DataFrame(columns=a_row.index)
    a_row_df = a_row_df.append(a_row, ignore_index=True)

    x_new, y_new = fraud_pkl.clean_data(a_row_df)


    y_pred = fraud_pkl.mod.predict(x_new)
    print(y_pred)

    return y_pred

if __name__ == '__main__':
    df = pd.read_json('data/data.json')
    with open('model.pkl', 'rb') as f:
        fraud_pkl = pickle.load(f)
    predict_stuff(fraud_pkl)
