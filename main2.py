import pandas as pd

###############################
# fruit dataset
###############################
fruits = pd.read_csv('fruit_data_with_colors.txt', sep='\t')
fruits.head()

Fruit_k = fruits[fruits['fruit_label'] == 2]
X = Fruit_k[['height', 'width']]
y = Fruit_k['mass']

lookup_fruit_name = dict(zip(Fruit_k.fruit_label.unique(), Fruit_k.fruit_name.unique()))
print(lookup_fruit_name)

# plotting the data
from matplotlib import pyplot as plt
plt.subplot(2, 1, 1)
plt.scatter(X['height'], y, marker='o', color='blue', s=12)
plt.xlabel('height')
plt.ylabel('mass')

plt.subplot(2, 1, 2)
plt.scatter(X['width'], y, marker='o', color='blue', s=12)
plt.xlabel('width')
plt.ylabel('mass')
plt.show()

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a linear model : Linear regression (aka ordinary least squares)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

# Estimate the accuracy of the classifier on future data, using the test data
# score = 1-relative score
# R^2(y, hat{y}) = 1 - {sum_{i=1}^{n} (y_i - hat{y}_i)^2}/{sum_{i=1}^{n} (y_i - bar{y})^2}
##########################################################################################
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# Use the trained linear repression model to predict a new, previously unseen object
# first example: a small fruit with width 7.5 cm, height 7.3 cm, mass in kg
fruit_prediction = lr.predict([[7.5, 7.3]])
print("mass: {}".format(fruit_prediction[0]))

#########################################################################
#########################################################################
#More complicated data : Boston housing dataset
#https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html
#Samples total 506
#Dimensionality 13
#Features real, positive
#Targets real 5. - 50. *$1000 = price
#########################################################################
from sklearn.datasets import load_boston
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a linear model : Linear regression (aka ordinary least squares)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("lr.coef_: {}".format(lr.coef_))
print("sum lr.coef_^2: {}".format(sum(lr.coef_*lr.coef_)))
print("lr.intercept_: {}".format(lr.intercept_))

# Estimate the accuracy of the classifier on future data, using the test data
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation: default 5-fold cross validation cv=5
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(lr, X, y, cv=10)

# plotting the data
from matplotlib import pyplot as plt
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.scatter(y, predicted, edgecolors=(0, 0, 0))
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax1.set_xlabel('Measured')
ax1.set_ylabel('Predicted')

# Leave one out: Provides train/test indices to split data in train/test sets. Each
# sample is used once as a test set (singleton) while the remaining samples form the training set.
# n= the number of samples
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(X)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
predicted = []
measured = []
for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr.fit(X_train, y_train)
    predicted.append(lr.predict(X_test)[0])
    measured.append(y_test[0])

ax2.scatter(measured, predicted, edgecolors=(0, 0, 0))
ax2.plot([min(measured), max(measured)], [min(measured), max(measured)], 'k--', lw=4)
ax2.set_xlabel('Measured')
ax2.set_ylabel('Predicted')
plt.show()

###############################################################
# Ridge regression --- a more stable model
# In ridge regression, the coefficients (w) are chosen not only so that they predict well on the training
# data, but also to fit an additional constraint. We also want the magnitude of coefficients
# to be as small as possible; in other words, all entries of w should be close to zero.
# This constraint is an example of what is called regularization. Regularization means explicitly
# restricting a model to avoid overfitting.
# minimizing ||y - Xw||^2_2 + alpha * ||w||^2_2
# Note: the smaller alpha = the less restriction.
###############################################################
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=10).fit(X_train, y_train)
print("ridge.coef_: {}".format(ridge.coef_))
print("sum ridge.coef_^2: {}".format(sum(ridge.coef_*ridge.coef_)))
print("ridge.intercept_: {}".format(ridge.intercept_))
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

###############################################################
# Lasso regression --- a more stable model
# In lasso regression, the coefficients (w) are chosen not only so that they predict well on the training
# data, but also to fit an additional constraint. We also want the magnitude of coefficients
# to be as small as possible; in other words, all entries of w should be close to zero.
# This constraint is an example of what is called regularization. Regularization means explicitly
# restricting a model to avoid overfitting.
# minimizing ||y - Xw||_2 + alpha * ||w||_2
# Note: the smaller alpha = the less restriction.
###############################################################
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
print("lasso.coef_: {}".format(lasso.coef_))
print("sum lasso.coef_^2: {}".format(sum(lasso.coef_*lasso.coef_)))
print("lasso.intercept_: {}".format(lasso.intercept_))
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))