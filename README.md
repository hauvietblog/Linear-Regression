# [Machine Learning cơ bản](https://machinelearningcoban.com/about/)
## Bài 1: [Linear Regression](https://machinelearningcoban.com/2016/12/28/linearregression/)
### 1. Bài toán
Xét ánh xạ  

$$\begin{aligned} 
g: \mathbb{R}^m &\to \mathbb{R} \\\\ (x_1,x_2,\dots,x_m) &\mapsto g(x_1,x_2,\dots,x_m)=y 
\end{aligned}$$

Giả sử có n điểm dữ liệu trong không gian m-chiều $(x_1^{(i)},x_2^{(i)},\dots,x_m^{(i)}), i=1 \dots n$, và $g(x_1^{(i)},x_2^{(i)},\dots,x_m^{(i)}) = y^{(i)}$  
Xét ánh xạ 

$$\begin{aligned} 
f: \mathbb{R}^{m+1} \times \mathbb{R}^{m +1} &\to \mathbb{R} \\\\ (\mathbf{x},\mathbf{w}) &\mapsto f(\mathbf{x},\mathbf{w}) \approx y 
\end{aligned}$$

chúng ta cần tìm một hàm số $f$ sao cho
$$f(\mathbf{x},\mathbf{w})=w_1x_1+w _2x_2+\dots+w_mx_m+w_0 \approx y ~~~~~(1)$$

Giả sử hàm $f$ tuyến tính và phụ thuộc vào các tham số $w_i, i= 0,\dots,m$. Đặt $\mathbf{w}=(w_0,w_1,\dots,w_m)^T$ và $\bar\{\mathbf{x}\}=(1,x_1,\dots,x_m)$ khi đó  phương trình (1) viết lại dưới dạng:
$$y \approx \bar{\mathbf{x}}\mathbf{w}$$ 
Nội dung của phương pháp là tìm các giá trị của th am số $\mathbf{w}$ sao cho  biểu thức sau đạt cực tiểu
$$\mathcal{L}(\mathbf{w})=\frac{1}{2}\sum_{i=1}^n(y_{i}-\bar{\mathbf{x}}_{i}\mathbf{w})^2= \frac{1}{2} ||\mathbf{y} - \mathbf{\bar{X}}\mathbf{w} ||_2^2 $$   
###  
Trong đó $\mathbf{y}=(y_{1},y_{2},\dots,y_{n})^T,\mathbf{\bar{X}} = [\mathbf{\bar{x}_1}\mathbf{\bar{x}_2}; \dots \mathbf{\bar{x}}_n]={\left\lbrack \matrix{1&x_1^{(1)}&x_2^{(1)}&\dots&x_m^{(1)} \cr 1&x_1^{(2)}&x_2^{(2)}&\dots&x_m^{(2)} \cr \dots&\dots&\dots&\dots&\dots \cr 1&x_1^{(n)}&x_2^{(n)}&\dots&x_m^{(n)} } \right\rbrack}$
### 2. Nghiệm của bài toán
Đạo hàm theo $\mathbf{w}$ của hàm $\mathcal{L}(\mathbf{w})$
$${\nabla_{\mathbf{w}}}\mathcal{L}(\mathbf{w})=\frac{1}{2}\nabla{||\mathbf{y}-\bar{\mathbf{X}}\mathbf{w}||}_2^2=\frac{1}{2}\nabla[(\mathbf{y}-\bar{\mathbf{X}}\mathbf{w})^T(\mathbf{y}-\bar{\mathbf{X}}\mathbf{w})]=\bar{\mathbf{X}}^T(\bar{\mathbf{X}}\mathbf{w}-\mathbf{y})$$\
Để tìm các giá trị của tham số $\mathbf{w}$ ta giải phương trình đạo hàm bằng 0\
$$\mathbf{\bar{X}}^T\mathbf{\bar{X}}\mathbf{w} = \mathbf{\bar{X}}^T\mathbf{y} \triangleq \mathbf{b}~~~~(2)$$
Nếu ma trận vuông $\mathbf{A} \triangleq \mathbf{\bar{X}}^T\mathbf{\bar{X}}$ khả nghịch thì phương trình $(2)$ có nghiệm duy nhất: $\mathbf{w} = \mathbf{A}^{-1}\mathbf{b}$
Nếu ma trận $\mathbf{A}$ không khả nghịch thì phương trình $(2)$ có nghiệm 
$$\mathbf{w} = \mathbf{A}^{\dagger}\mathbf{b} = (\mathbf{\bar{X}}^T\mathbf{\bar{X}})^{\dagger} \mathbf{\bar{X}}^T\mathbf{y}$$
### 3. Ví dụ trên python
#### 3.1. Bài toán
Chúng ta có 1 bảng dữ liệu về năm và dân số nước ta như dưới đây:
###
$$\begin{array}{|c|c|}
\hline
\textbf{Year} & \textbf{Population}\\
\hline
\text{1950} & \text{25109200}\\
\hline
\text{1960} & \text{32718461}\\
\hline
\text{1970} & \text{41928849}\\
\hline
\text{1980} & \text{52968270}\\
\hline
\text{1990} & \text{66912613}\\
\hline
\text{2000} & \text{79001142}\\
\hline
\text{2010} & \text{87411012}\\
\hline
\text{2020} & \text{96648685}\\
\hline
\end{array}$$
###
#### 3.2. Hiển thị dữ liệu trên đồ thị
````python
import matplotlib.pyplot as plt
import numpy as np
X = np.array([[1950,1960,1970,1980,1990,2000,2010,2020]]).T
y = np.array([[25109200,32718461,41928849,52968270,66912613,79001142,87411012,96648685]]).T
plt.plot(X,y,"go")
plt.title('Vietnam Population 1950-2020')
plt.xlabel('Year')
plt.ylabel('Population')
plt.show()
````
###
![bieudodsvn](https://user-images.githubusercontent.com/72483300/216109556-8fc57231-3181-4006-b52b-25661e3282d1.png)
###
$$\text{năm} = w_1*(\text{dân số}) + w_0$$
#### 3.3. Nghiệm theo công thức
````python
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
````
![linear_regression](https://user-images.githubusercontent.com/72483300/216771842-023f755e-9533-410a-be4e-cd7e1f8890c0.png)
### 4. Tài liệu tham khảo
  1. [Linear Regression - Machine Learning cơ bản](https://machinelearningcoban.com/2016/12/28/linearregression/#-bai-toan)
  2. [Bình phương tối thiểu - Wikipedia](https://vi.wikipedia.org/wiki/B%C3%ACnh_ph%C6%B0%C6%A1ng_t%E1%BB%91i_thi%E1%BB%83u)
  3. [Dân số việt nam 1950-2020](https://www.macrotrends.net/countries/VNM/vietnam/population)
