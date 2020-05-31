# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Multiple Linear Regression
# %% [markdown]
# ## Importing the libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# ## Importing the dataset

# %%
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
pd.DataFrame(X)

# %%
# %% [markdown]
# ## Encoding categorical data

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
pd.DataFrame(X, columns=["California", "Florida","New York", "R&D Spend", "Administration", "Marketing Spend"])

# %% [markdown]
# ## Splitting the dataset into the Training set and Test set

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

pd.DataFrame(X_train, y_train)
# %% [markdown]
# ## Training the Multiple Linear Regression model on the Training set

# %%
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# %% [markdown]
# ## Predicting the Test set results

# %%
# y_pred = regressor.predict(X_test)
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# pd.DataFrame
# %%
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
# np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
pd.DataFrame(X_test, y_pred)
# %%
columns = ["California", "Florida","New York", "R&D Spend", "Administration", "Marketing Spend"]
pd.DataFrame(X_test, columns=columns).assign(y_data=y_test).assign(y_pred=y_pred)