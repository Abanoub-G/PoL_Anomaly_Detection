import numpy as np
import pandas as pd 

X_outliers = np.random.uniform(low=3.5, high = 6, size=(3000, 100))

pd.DataFrame(X_outliers).to_csv("X_outliers.csv")


X_nomral = np.random.uniform(low = 0, high = 4,size=(20000, 100) )
pd.DataFrame(X_nomral).to_csv("X_normal.csv")