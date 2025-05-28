
### Goal

- make $\beta_k$'s comparable
- numerical stability in $(\mathbf{X}^T\mathbf{X})^{-1}$

### Standardization

$$
\frac{Y_i-\bar Y}{S_y}\quad\quad \frac{X_{i,k}-\bar{X_k}}{S_k}
$$

### Correlation Transform

$$
Y_i^*=\frac{1}{\sqrt{n-1}}\frac{Y_i-\bar Y}{S_y}\quad\quad X_{i,k}^*=\frac{1}{\sqrt{n-1}}\frac{X_{i,k}-\bar{X_k}}{S_k}
$$
$$
Y_i^*=\beta_1^*X_{i,1}^*+\beta_2^*X_{i,2}^*+\cdots+\beta_{p-1}^*X_{i,p-1}^*+\varepsilon_i^*
$$
Notice that the model does not have no intercept term $\beta_0^*=0$.

We can calculate $\mathbf{X}^T\mathbf{X}=\mathbf{r}_{xx}=$ sample correlation matrix between $\mathbf{X}_i$'s.
$$
	[\mathbf{r}_{xx}]_{ij}=r_{ij}=\frac{1}{n-1}\frac{\sum_{k=1}^n(X_{ki}-\bar X_i)(X_{kj}-\bar X_j)}{S_i\times S_j}\qquad |r_{ij}|\leq1
$$
$$
\mathbf{X}^T\mathbf{Y}=\mathbf{r}_{XY}=\begin{bmatrix}r_{y1}\\\vdots\\r_{y,p-1}\end{bmatrix}
$$
where $r_{y,j}=$ sample correlation btw $Y$ and $X_j$.
$$
\begin{array}{}
\mathbf{b}^*&=&\hat\beta^*=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}\\
&=&\mathbf{r}_{xx}^{-1}\mathbf{r}_{XY}\\
&=& \begin{bmatrix}
b_1^*\\\vdots\\b_{p-1}^*
\end{bmatrix}
\end{array}
$$
We have
$$
b_k=\frac{S_Y}{S_k}\times b_k^*\qquad b_0=\bar Y-b_1\bar X_1-\cdots-b_{p-1}\bar X_{p-1}
$$
---