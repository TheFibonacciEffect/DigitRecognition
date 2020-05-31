#%%
import sklearn
import pandas as pd

#%%
data = pd.read_csv("Social_Network_Ads.csv")
X = data.iloc[:,:-1]
y = data.iloc[:, :1]

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# %%
