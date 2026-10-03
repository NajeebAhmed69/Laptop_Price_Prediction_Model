import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR

df=pd.read_csv('laptop_data.csv')


df.drop(columns=['Unnamed: 0'],inplace=True)
df['Ram']=df['Ram'].str.replace('GB','').astype('int32')
df['Weight']=df['Weight'].str.replace('kg','').astype('float32')


df['TouchScreen']=df['ScreenResolution'].apply(lambda x:1 if 'TouchScreen' in x else 0)
df['Ips']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

split_res=df['ScreenResolution'].str.split('x',n=1,expand=True)
df['X_res']=split_res[0].str.extract(r'(\d+)').astype('int')
df['Y_res']=split_res[1].astype('int')



df['ppi'] = (
    ((df['X_res'] ** 2 + df['Y_res'] ** 2) ** 0.5) / df['Inches']
).astype('float32')
df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

df['Cpu Name']=df['Cpu'] .apply(lambda x:" ".join(x.split()[0:3]))

def fetch_processor(text):
    if text =="Intel Core i7" or text=="Intel Core i5" or text=="Intel Core i3":
        return text
    else:
        if text.split()[0]=="Intel":
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


df['Cpu Brand']=df['Cpu Name'].apply(fetch_processor)

df.drop(columns=['Cpu','Cpu Name'], inplace=True)



df['Memory'] = df['Memory'].astype(str).str.replace(r'\.0', '', regex=True)
df['Memory'] = df['Memory'].str.replace('GB', '')
df['Memory'] = df['Memory'].str.replace('TB', '000')

mem_split = df['Memory'].str.split('+', n=1, expand=True)
df['first'] = mem_split[0].str.strip()

# Check if there is a second drive; handle missing values without chained inplace=True
if 1 in mem_split.columns:
    df['second'] = mem_split[1].str.strip().fillna('0')
else:
    df['second'] = '0'

for layer, col in [('first', '1'), ('second', '2')]:
    # Use str(x) to prevent float NaN crashes
    df[f'Layer{col}HDD'] = df[layer].apply(lambda x: 1 if 'HDD' in str(x) else 0)
    df[f'Layer{col}SSD'] = df[layer].apply(lambda x: 1 if 'SSD' in str(x) else 0)
    df[f'Layer{col}Hybrid'] = df[layer].apply(lambda x: 1 if 'Hybrid' in str(x) else 0)
    df[f'Layer{col}Flash_Storage'] = df[layer].apply(lambda x: 1 if 'Flash Storage' in str(x) else 0)
    
    # Extract numerical disk sizes
    df[layer] = df[layer].astype(str).str.extract(r'(\d+)').fillna(0).astype('int')

df['HDD'] = (df['first'] * df['Layer1HDD'] + df['second'] * df['Layer2HDD']).astype('int')
df['SSD'] = (df['first'] * df['Layer1SSD'] + df['second'] * df['Layer2SSD']).astype('int')
df['Hybrid'] = (df['first'] * df['Layer1Hybrid'] + df['second'] * df['Layer2Hybrid']).astype('int')
df['Flash_Storage'] = (df['first'] * df['Layer1Flash_Storage'] + df['second'] * df['Layer2Flash_Storage']).astype('int')

# Drop intermediate helper columns
cols_to_drop = [
    'first', 'second',
    'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage',
    'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2Flash_Storage',
    'Memory', 'Hybrid', 'Flash_Storage'
]
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

df['Gpu Brand']=df['Gpu'].apply(lambda x: x.split()[0])
df = df[df['Gpu Brand'] != 'ARM']
df.drop(columns=['Gpu'],inplace=True)

def categorize_os(val):
    if 'Windows' in val:
        return 'Windows'
    elif 'Mac' in val:
        return 'Mac'
    else:
        return 'Others/NO OS/Linux'


df['os']=df['OpSys'].apply(categorize_os)
df.drop(columns=['OpSys'], inplace=True)


X=df.drop(columns=['Price'])
y=np.log(df['Price'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
# Categorical column indices for OneHotEncoder:
# Company(0), TypeName(1), Cpu_brand(7), Gpu_brand(10), os(11)
cat_indices = [0, 1, 7, 10, 11]
preprocessor=ColumnTransformer(
    transformers=[
        ('col_tnf',OneHotEncoder(sparse_output=False,drop='first',handle_unknown='ignore'),cat_indices)
    ],
    remainder='passthrough',
)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10.0),
    'Lasso Regression': Lasso(alpha=0.001),
    'KNN Regressor': KNeighborsRegressor(n_neighbors=3),
    'Support Vector Regressor': SVR(kernel='rbf', C=10000, epsilon=0.1),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=15
    ),
    'Extra Trees': ExtraTreesRegressor(
        n_estimators=100, random_state=42, max_depth=15
    ),
    'AdaBoostRegressor':AdaBoostRegressor(n_estimators=500,random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=500, random_state=42
    ),
}


print(f"{'Model':<25} | {'R2 Score':<10} | {'MAE (log scale)':<15}")
print('-' * 55)

best_score = -1.0
best_model_name = None
best_pipeline = None

for name, regressor in models.items():
  pipeline = Pipeline([('step1', preprocessor), ('step2', regressor)])
  pipeline.fit(X_train, y_train)
  y_pred = pipeline.predict(X_test)

  score = r2_score(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  print(f'{name:<25} | {score:.4f}     | {mae:.4f}')

  if score > best_score:
    best_score = score
    best_model_name = name
    best_pipeline = pipeline

print(f'\nSelected Best Estimator: {best_model_name} (R2: {best_score:.4f})')

with open('df.pkl', 'wb') as f:
  pickle.dump(df, f)

with open('pipeline.pkl', 'wb') as f:
  pickle.dump(best_pipeline, f)

print("Exported 'df.pkl' and 'pipeline.pkl' successfully.")












