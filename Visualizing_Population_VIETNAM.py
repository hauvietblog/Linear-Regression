import matplotlib.pyplot as plt
import numpy as np
X = np.array([[1950,1960,1970,1980,1990,2000,2010,2020]]).T
y = np.array([[25109200,32718461,41928849,52968270,66912613,79001142,87411012,96648685]]).T
def Visualizing_Population_VIETNAM(X,y):
	plt.plot(X,y,"go")
	plt.plot()
	plt.title('Vietnam Population 1950-2020')
	plt.xlabel('Year')
	plt.ylabel('Population')
	plt.show()
if __name__ == "__main__":
	Visualizing_Population_VIETNAM(X,y)