import pandas as pd
from sklearn.ensemble import IsolationForest

# load data
data = pd.read_csv('data.csv')

# select features
features = data[['transactions', 'failed_logins', 'time_spent', 'location_changes']]

# train model
model = IsolationForest(contamination=0.3)
model.fit(features)

# predict risk
data['anomaly'] = model.predict(features)

# convert to readable output
data['risk'] = data['anomaly'].apply(lambda x: 'High Risk' if x == -1 else 'Normal')

print(data[['user_id', 'risk']])
