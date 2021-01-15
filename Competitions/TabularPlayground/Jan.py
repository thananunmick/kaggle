import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

def load_train_set():
    train_path = os.path.join("datasets", "Jan", "train.csv")
    return pd.read_csv(train_path)

train = load_train_set()
train_set, val_set = train_test_split(train, test_size=0.3, random_state=42)
X_train = train_set.drop(["id", "target"], axis=1)
y_train = train_set["target"].copy()
X_val = val_set.drop(["id", "target"], axis=1)
y_val = val_set["target"].copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Draw a histogram
train.hist(bins=100, figsize=(20, 15))

# Look at the correlations between each features
corr_matrix = train.corr()
print(corr_matrix["target"].sort_values(ascending=False))

# Draw a scatter matrix to see the correlation
attributes = ["target", "cont7", "cont2", "cont3"]
# scatter_matrix(train[attributes], figsize=(12, 8))
# plt.show()

# Check for any Nan values
def check_nan():
    for i in range(len(X_train_scaled.columns)):
        column = X_train_scaled.iloc[:, i]
        isnull = False
        if column.isnull().values.any():
            isnull = True
    
        print("Column " + str(i) + str(isnull))

# Use Stochastic Gradient Descent to fit the data
def run_sgd_reg():
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
    return sgd_reg.fit(X_train_scaled, y_train)

# reggressor = run_sgd_reg()

# Use Lienar Regression to fit the data
def run_lin_reg():
    lin_reg = LinearRegression()
    return lin_reg.fit(X_train_scaled, y_train)

# reggressor = run_lin_reg()

# Use SVM Regression to fit the data
def run_svm_reg():
    # grid_param = {"epsilon": [0, 0.5, 1, 1.5, 2], "C": [1, 2, 3, 4, 5]}
    # svm_reg = LinearSVR(max_iter=3000)
    # grid = GridSearchCV(svm_reg, grid_param)
    # return grid.fit(X_train_scaled, y_train)
    svm_reg = SVR(kernel="poly", degree=2)
    return svm_reg.fit(X_train_scaled, y_train)

def train():
    # reggressor = run_sgd_reg()
    # reggressor = run_lin_reg()
    reggressor = run_svm_reg()
    return reggressor

reggressor = train()

print("Finished Training")
train_mse = cross_val_score(reggressor, X_train_scaled, y_train, scoring="neg_mean_squared_error", cv=5)
val_mse = cross_val_score(reggressor, X_val, y_val, scoring="neg_mean_squared_error", cv=5)
print("RMSE_Train_Set: " + str(np.sqrt(-train_mse)))
print("RMSE_Val_Set: " + str(np.sqrt(-val_mse)))

# plt.show()

# #######################################################
# # Running Test set

# # Load Test set
# def load_test_set():
#     test_path = os.path.join("datasets", "Jan", "test.csv")
#     return pd.read_csv(test_path)

# test = load_test_set()
# X_test = test.drop(["id"], axis=1)
# id_test = test["id"].copy()
# # prediction = cross_val_predict(reggressor, X_test, cv=5, method="predict")
# prediction = reggressor.predict(X_test)
# prediction = prediction.reshape(len(prediction), 1)
# prediction = np.c_[id_test, prediction]
# print(prediction)
# np.savetxt("Prediction.csv", prediction, delimiter=",", fmt="%i,%.1f")