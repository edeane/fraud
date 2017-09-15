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

def smote_sample(X_train, y_train, seed=321):
    sm = SMOTE(random_state=seed)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    return X_train_res, y_train_res


class Fraud(object):
    def __init__(self):
        pass

    def get_modes(self, df):
        self.fill_dict = {}
        fill_per = df.count() / max(df.count()) * 100
        less_than_100 = fill_per[fill_per<100].index
        for col in less_than_100:
            self.fill_dict[col] = df[col].mode()[0]
        self.good_country_list = df.country.value_counts()[df.country.value_counts() > 500].index
        self.good_email_list = df.email_domain.value_counts()[df.email_domain.value_counts() > 150].index
        fill_per = df.count() / max(df.count()) * 100
        self.less_than_100 = fill_per[fill_per<100].index


    def clean_data(self, df, calc_modes=False):

        if calc_modes:
            self.get_modes(df)

        for col in self.less_than_100:
            df[col].fillna(self.fill_dict[col], inplace=True)

        def fraud_check(row):
            if 'fraud' in row:
                return 1
            else:
                return 0

        def good_country(x):
            if x in self.good_country_list:
                return 1
            else:
                return 0


        def good_email(x):
            if x in self.good_email_list:
                return 1
            else:
                return 0

        def quant_sold(x):
            q_sold = 0
            for j in x:
                q_sold+=j['quantity_sold']
            return q_sold

        #df = pd.get_dummies(df, columns=['currency'])
        df['fraud'] = 0
        if calc_modes:
            df['fraud'] = df['acct_type'].apply(fraud_check)
        df['description_len'] = df['description'].apply(lambda x: len(x))
        df['listed'] = df['listed'].map({'y': 1, 'n': 0})
        df['len_pp'] = df['previous_payouts'].apply(lambda x: len(x))
        df['good_email'] = df['email_domain'].apply(good_email)
        df['good_country'] = df['country'].apply(good_country)
        df['len_tt'] = df['ticket_types'].apply(lambda x: len(x))
        df['q_sold'] = df['ticket_types'].apply(quant_sold)

        fraud_corr = abs(df.corr()['fraud']).sort_values(ascending=False)
        good_columns = fraud_corr[fraud_corr > .08].index

        if calc_modes:
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            num_df = df.select_dtypes(include=numerics)
            self.num_columns = num_df.columns

        clean_df = df[self.num_columns]

        x = clean_df.drop('fraud', axis=1)
        y = clean_df['fraud']

        return x, y

    def model(self, x, y):
        x_train = x.copy()
        y_train = y.copy()
        x_test = x.copy()
        y_test = y.copy()
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)
        mod = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=1000))
        mod.fit(x_train, y_train)
        print(x_train.shape)

        print(mod.steps[1][1].__class__.__name__)
        print('model score: ', mod.score(x_test, y_test))
        y_pred = mod.predict(x_test)
        y_pred = pd.Series(y_pred, name='pred')
        print('recall', recall_score(y_test, y_pred))
        print('precision', precision_score(y_test, y_pred))
        print(pd.crosstab(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print('----------------------------')

        probas = mod.predict_proba(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        print(mod.steps[1][1].__class__.__name__, 'auc', roc_auc)
        plt.plot(fpr, tpr, lw=3, label=mod.steps[1][1].__class__.__name__)
        plt.legend(loc='best')

        plt.show()

        feat_impor = mod.steps[1][1].feature_importances_
        x_feat = pd.DataFrame({'cols': x.columns, 'feat_impo': feat_impor})
        print(mod.steps[1][1].__class__.__name__, '\n', x_feat.sort_values(by='feat_impo', ascending=False))

        self.mod = mod

        pass

def make_stuff():
    fraud = Fraud()
    x, y = fraud.clean_data(df, calc_modes=True)
    fraud.model(x, y)

    with open('model.pkl', 'wb') as f:
        pickle.dump(fraud, f)
    pass


if __name__ == '__main__':
    df = pd.read_json('data/data.json')
    make_stuff()








    pass
