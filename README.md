# [Machine Learning cơ bản](https://machinelearningcoban.com/about/)
## Bài 1: [Linear Regression](https://machinelearningcoban.com/2016/12/28/linearregression/)
### 1. Bài toán
Giả sử có n điểm dữ liệu trong không gian n-chiều, chúng ta cần tìm một hàm $f$ thỏa mãn 
$$f(\mathbf{x}) \approx y$$
$$f(\mathbf{x})=w_1x_1+w_2x_2+\dots+w_nx_n+w_0 ~~~~~(1)$$
Giả sử hàm $f$ tuyến tính và phụ thuộc vào các tham số $w_i, i= 0,\dots,n$. Đặt $\mathbf{w}=(w_0,w_1,\dots,w_n)^T$ và $\bar{\mathbf{x}}=(1,x_1,\dots,x_n)$ khi đó phương trình (1) viết lại dưới dạng:
$$y \approx \bar{\mathbf{x}}\mathbf{w}$$
Nội dung của phương pháp là tìm các giá trị của tham số $\mathbf{w}$ sao cho biểu thức sau đạt cực tiểu
$$\mathcal{L}(\mathbf{w})=\frac{1}{2}\sum_{i=1}^n(y_i-\bar{\mathbf{x}}\mathbf{w})^2= \frac{1}{2} ||\mathbf{y} - \mathbf{\bar{X}}\mathbf{w} ||_2^2 $$
Trong đó $\mathbf{y}=(y_1,y_2,\dots,y_n)^T,\bar{\mathbf{X}}=[\bar{x_1;x_2;\dots;x_n}]$
### 2. Nghiệm của bài toán
Đạo hàm theo $\mathbf{w}$ của hàm $\mathcal{L}(\mathbf{w})$
$${\nabla_{\mathbf{w}}}\mathcal{L}(\mathbf{w})=\frac{1}{2}\nabla{||\mathbf{y}-\bar{\mathbf{X}}\mathbf{w}||}_2^2=\frac{1}{2}\nabla[(\mathbf{y}-\bar{\mathbf{X}}\mathbf{w})^T(\mathbf{y}-\bar{\mathbf{X}}\mathbf{w})]=\bar{\mathbf{X}}^T(\bar{\mathbf{X}}\mathbf{w}-\mathbf{y})$$\
Để tìm các giá trị của tham số $\mathbf{w}$ ta giải phương trình đạo hàm bằng 0\
$$\mathbf{\bar{X}}^T\mathbf{\bar{X}}\mathbf{w} = \mathbf{\bar{X}}^T\mathbf{y} \triangleq \mathbf{b}~~~~(2)$$
Nếu ma trận vuông $\mathbf{A} \triangleq \mathbf{\bar{X}}^T\mathbf{\bar{X}}$ khả nghịch thì phương trình $(2)$ có nghiệm duy nhất: $\mathbf{w} = \mathbf{A}^{-1}\mathbf{b}$
Nếu ma trận $\mathbf{A}$ không khả nghịch thì phương trình $(2)$ có nghiệm 
$$\mathbf{w} = \mathbf{A}^{\dagger}\mathbf{b} = (\mathbf{\bar{X}}^T\mathbf{\bar{X}})^{\dagger} \mathbf{\bar{X}}^T\mathbf{y}$$
### 3. Ví dụ trên python
