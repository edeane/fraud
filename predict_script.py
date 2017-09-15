
import json
import requests
import time
from datetime import datetime
import pickle
from model import Fraud
import pandas as pd
from model import Fraud

# predictions = pd.DataFrame(columns=['approx_payout_date', 'body_length', 'channels', 'country', 'currency',
#        'delivery_method', 'description', 'email_domain', 'event_created',
#        'event_end', 'event_published', 'event_start', 'fb_published', 'gts',
#        'has_analytics', 'has_header', 'has_logo', 'listed', 'name',
#        'name_length', 'num_order', 'num_payouts', 'object_id', 'org_desc',
#        'org_facebook', 'org_name', 'org_twitter', 'payee_name', 'payout_type',
#        'previous_payouts', 'sale_duration', 'sale_duration2', 'show_map',
#        'ticket_types', 'user_age', 'user_created', 'user_type',
#        'venue_address', 'venue_country', 'venue_latitude', 'venue_longitude',
#        'good_email', 'good_country', 'len_tt', 'q_sold', 'prediction'
#        'venue_name', 'venue_state', 'fraud', 'description_len', 'len_pp',

url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
with open('model.pkl', 'rb') as fr:
    fraud_pkl = pickle.load(fr)

with open('predictions.pkl', 'rb') as fr:
    predictions = pickle.load(fr)

while True:
    content = requests.get(url).content
    content = json.loads(content)
    df = pd.DataFrame.from_dict(content, orient='index').T
    if df['name'].values[0] not in predictions['name'].values:
        x_web, y_web = fraud_pkl.clean_data(df)
        y_pred = fraud_pkl.mod.predict(x_web)
        if y_pred[0] == 0:
            df['prediction'] = 'No'
        elif y_pred[0] == 1:
            df['prediction'] = 'Yes'
        predictions = predictions.append(df, ignore_index=True)
        print(predictions.shape)
        with open('predictions.pkl', 'wb') as pr:
            pickle.dump(predictions, pr)
        time.sleep(1)
