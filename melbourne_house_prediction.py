import pandas as pd
pd.plotting.register_matplotlib_converters()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

houses = pd.read_csv('./melb_data.csv')

y = houses.Price
X = houses.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# preprocces for numerical variable
numerical_transformer = SimpleImputer(strategy='constant')

#preprocces for catigorical variable
catigorical_transformer = Pipeline(steps=[
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('OH',OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('numbers',numerical_transformer,numerical_cols),
    ('catigorical',catigorical_transformer,categorical_cols)
])

model = RandomForestRegressor(n_estimators=100, random_state=0)

pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model)
])

#Time for prediction
# pipeline.fit(X_train,y_train)
#
# preds = pipeline.predict(X_valid)
#
# score = mean_absolute_error(y_valid, preds)
# print('MAE:', score)

# here i will use cross-validation
from sklearn.model_selection import cross_val_score

cross_val_accurecy = cross_val_score(pipeline,X,y,cv=5)

print(cross_val_accurecy.mean())

# output = pd.DataFrame({'predicted value':preds , 'real value':y_valid})
# output.to_csv('./submission_test',index=False)