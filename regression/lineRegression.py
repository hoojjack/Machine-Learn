import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


X  = np.linspace(1,6,100)

y = 0.3 * X + 2 + np.random.normal(0,1,100)
print ('X = ',X)
print ('y = ',y)
y = y.reshape(-1,1)

X = X.reshape(-1,1)

print ('X = ', X)

print ('y = ',y)




from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)

print ("inter = ", linreg.intercept_)
print ("coef = ",  linreg.coef_)

from sklearn.linear_model import RidgeCV

ridgecv = RidgeCV(alphas=[0.01,0.1,0.5,1,3,5,7,10,20,100])
ridgecv.fit(X_train, y_train)

y_ridge_pred = ridgecv.predict(X_test)

plt.plot(X,y,'r*',label='origianl data')
plt.ylim(y.min()-1,y.max()+1)

plt.plot(X_test,y_pred,'bo',label='line predict data')

plt.plot(X_test,y_ridge_pred,'yo',label='ridge predict data')

plt.legend()
plt.show()
plt.savefig("lg1.png")
