from Visualizing_Population_VIETNAM import X,y
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
one = np.ones([X.shape[0],1])
Xbar = np.concatenate((one,X), axis = 1)
A = np.dot(Xbar.T,Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A),b)
print('w = ',w)
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar,y)
print('solution by scikit-learn :',regr.coef_)
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(1950, 2020)
y0 = w_0 + w_1*x0
plt.plot(X.T, y.T, 'go')     
plt.plot(x0, y0)   
plt.title('Vietnam Population 1950-2020')        
plt.xlabel('Year')
plt.ylabel('Population')
plt.show()
