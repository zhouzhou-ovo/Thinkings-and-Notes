
$$
Y_{1,i}=\beta_0+\beta_1Y_{2,i}+\varepsilon_i
$$
where $Y_1,Y_2$ are both random variable (In SLR model, $X_i$ are considered fixed/given)

In CMR, $(Y_1,Y_2)\thicksim MVN(\mu,\Sigma)$ , where $\mu=(\mu_1,\mu_2)^T$ , $\Sigma$ is covariance matrix, then we have
$$
Y_1|Y_2\thicksim N(\alpha+\beta Y_2,\sigma^2)
$$
where $\alpha=\mu_1-\mu_2\rho\frac{\sigma_1}{\sigma_2},\,\beta=\rho\frac{\sigma_1}{\sigma_2},\,\sigma^2=\sigma_1^2(1-\rho^2)$ . We can also get
$$
Y_1|Y_2=y_2\thicksim N(\beta_0+\beta_1y_2,\sigma^2)
$$

### Hypothesis Testing

$$
H_0:\rho=0\quad vs\quad H_a:\rho\neq0
$$
>[!note] Definition(Correlation)
>$$
>\rho=\rho_{12}=corr(Y_1,Y_2)=\frac{cov(Y_1,Y_2)}{\sqrt{Var(Y_1)}\sqrt{Var(Y_2)}}
>$$

We have the estimator
$$
\hat{\rho_{12}}=r_{12}=\frac{\sum(Y_{i1}-\bar Y_1)(Y_{i2}-\bar Y_2)}{\sqrt{\sum_i(Y_{i1}-\bar Y_1)^2}\sqrt{\sum_i(Y_{i2}-\bar Y_2)^2}}
$$
called sample correlation coefficient or Pearson correlation coefficient or Pearson product moment coefficient.

Test statistics:
$$
t^*=\frac{r_{12}\sqrt{n-2}}{\sqrt{1-r_{12}^2}}\overset{H_0}\thicksim t_{n-2}
$$
We can prove that
$$
t^*=\frac{r_{12}\sqrt{n-2}}{\sqrt{1-r_{12}^2}}=\frac{b_1}{SE(b_1)}\quad\text{for }H_0:\beta_1=0
$$

### Interval Estimate for $\rho_{12}$

#### Idea

transform $\rho_{12}\in[-1,1]\rightarrow(-\infty,+\infty)$ and then use the CLT.

#### Fisher $z$-transform
$$
z'=\frac{1}{2}\log\frac{1+r_{12}}{1-r_{12}}
$$
And for large $n$, we have
$$
z'\thicksim N(\xi,\frac{1}{n-3})
$$
where $\xi=\frac{1}{2}\log\frac{1+\rho_{12}}{1-\rho_{12}}$ , and its value is the true value of $\xi$ (unbiased); $n$ is the sample size.

#### The procedure

- Find $r_{12}$ from data
- Transform $z'=\frac{1}{2}\log\frac{1+r_{12}}{1-r_{12}}\thicksim N(\xi,\frac{1}{n-3})$
- CI for $\xi$ : $z'\pm z(1-\alpha/2)\frac{1}{\sqrt{n-3}}$
- CI for $\rho_{12}$ : $\tanh\xi=\tanh(z'\pm z(1-\alpha/2)\frac{1}{\sqrt{n-3}})$
	$\tanh \xi =\frac{\sinh \xi}{\cosh \xi}=\frac{e^{2\xi}-1}{e^{2\xi}+1}$

---