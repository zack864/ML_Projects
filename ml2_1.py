import pandas as pd
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
app_rec = pd.read_csv("application_record.zip")
credit = pd.read_csv("credit_record.csv")
hru = credit[credit['STATUS'].isin(['2', '3', '4', '5'])]['ID'].unique()
target = pd.DataFrame({'ID': credit['ID'].unique()})
target['Target'] = target['ID'].apply(lambda x: 1 if x in hru else 0)
final_df = app_rec.merge(target, on='ID', how='inner')
final_df.drop('ID', axis=1, inplace=True)
encoders = {}
cat_cols = final_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    final_df[col] = le.fit_transform(final_df[col].astype(str))
    encoders[col] = le
X = final_df.drop('Target', axis=1)
y = final_df['Target']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)
with open('credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
joblib.dump(model, 'credit_model.joblib', compress=3)

print('1')
