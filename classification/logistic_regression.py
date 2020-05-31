
#%%
import pandas as pd
import sklearn
import numpy as np
# %%
csv= pd.read_csv("Social_Network_Ads.csv")
X = csv.iloc[:, :-1].values
y = csv.iloc[:, :1].values
X
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)
X_train
#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
# %%