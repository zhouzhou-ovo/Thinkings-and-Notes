
### Weighted Least Squares

>[!question]
>How do we fix non-constant variance?
>
>**Ans:**
>- Transform $Y$ via Box-Cox or Variance Stability Transform
>	- Destroy the linear relationship between $X$ and $Y$.
>
>
>- Weighted Least Squares
>	- Preserve linear relationship and address non-constant variance.

#### Case 1: Assume $\sigma_i^2$'s are given

$$
L(\beta)=[\prod_{i=1}^n(\frac{1}{2\pi\sigma_i^2})^{1/2}]\exp[-\frac{1}{2}\sum_{i=1}^n\frac{(y_i-\beta_0-\cdots-\beta_{P-1}x_{i,P-1})^2)}{\sigma_i^2}]
$$
Maximizing $L(\beta)$ w.r.t $\beta$ $\Updownarrow$ 

define $w_i\stackrel{\triangle}{=}\frac{1}{\sigma_i^2}$, we have
$$
Q_w(\beta)=\prod_{i=1}^n(\frac{w_i}{2\pi})^{-1/2}\exp[-\frac{1}{2}w_i(y_i-\beta_0-\cdots-\beta_{P-1}x_{i,P-1})^2]
$$
minimizing $Q_w(\beta)$ i.e. 
$$
Q_w(\beta)=(\mathbf{Y}-\mathbf{X}\beta)^T\mathbf{W}(\mathbf{Y}-\mathbf{X}\beta)
$$
where $\mathbf{W}=diag(w_1,\cdots,w_n),w_i=1/\sigma^2_i$ . We can also call $w_i$ as precision. (High $\sigma_i^2$ with low $w_i$; low $\sigma_i^2$ with high $w_i$)

We can get
$$
\text{Normal Equation:}(\mathbf{X}^T\mathbf{W}\mathbf{X})\mathbf{b}_w=\mathbf{X}^T\mathbf{WY}
$$
Then we have
$$
\mathbf{b}_w=(\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{Y}
$$
We can prove that
- $Var(\mathbf{b}_w)=(\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}$
- If $Var(\varepsilon_i)=\sigma^2$, then $\mathbf{b}_w=\hat{\mathbf{b}}^{OLS}$

**Properties of $\mathbf{b}_w$
$$
\begin{array}{rl}
\mathbf{b}_w &= \text{unbiased }(E(\mathbf{b}_w)=\beta)\\[8pt]

&= \text{min variance among unbiased linear estimators}\\[8pt]

&= \text{consistent}\quad(\text{i.e. }n\rightarrow \infty,\mathbf{b}_w\rightarrow\beta)
\end{array}
$$

#### Case 2: Variances are known up-to proportionality constant

e.g. $w_1=\frac{k}{2},w_2=\frac{k}{5},w_3=\frac{k}{2}$, i.e. we know $\frac{\sigma^2_i}{\sigma^2_j}$

Define $w_i=\frac{1}{k\sigma^2_i}$, where $k$ unknown, $\sigma^2_i$ given. We have
$$
\mathbf{W}(2)=\begin{bmatrix}
\frac{1}{k\sigma_1^2}& & \\
&\ddots&\\
&&\frac{1}{k\sigma_n^2}
\end{bmatrix}
$$

In this case, $Var(\mathbf{b}_w)=k(\mathbf{X}^T\mathbf{WX})^{-1}\Rightarrow s^2(\mathbf{b}_w)=MSE_w(\mathbf{X}^T\mathbf{WX})^{-1}$, where
$$
MSE_w\overset{\triangle}{=}\frac{1}{n-p}\sum_{i=1}^n\frac{1}{\sigma_i^2}e_i^2
$$
We can also prove that $\hat{k}_{MLE}=\frac{1}{n}\sum_{i=1}^nw_ie_i^2$. 

$n-p$ above is to make $MSE_w$ unbiased.

Case $2$ happens, for example, when response variable is an average of group of size $n_i$ :
$$
\text{response variable:} \bar{Y}_i=\frac{1}{n_i}\sum_{i=1}^{n_i}Y_{ij},\text{where }Y_{ij}\text{ has constant variance }k
$$
$\Rightarrow Var(\bar{Y}_i)=\frac{k}{n_i}=k\sigma_i^2$ where $k$ unknown, $n_i$ known.

#### Case 3: Variance $\sigma_i^2$ are unknown

$$
Var(\varepsilon_i)=E(\varepsilon_i^2)-E^2(\varepsilon_i)=E(\varepsilon_i^2)=\sigma^2_i
$$
so we have $\varepsilon_i^2=\sigma_i^2$, residual is the best guess of $\sigma_i^2$.

plot $|e_i| \text{ or }e_i^2$ vs $X_{ik}\text{ or }\hat{Y}_i$ , if plot has a megaphone shape, then regress $|e_i|$ or $e_i^2$ onto $X_{ik}$ or $\hat{Y}_i$
- fit $|e_i|$ vs $X_{ik}$ , fitted values $\hat{s}_i=\text{estimate of  SD }\sigma_i$ 
- fit $e_i^2$ vs $X_{ik}$ , fitted values $\hat{v}_i=\text{estimate of  Variance }\sigma_i^2$ 
- choose $w_i=\frac{1}{\hat{v}_i}$ or $\frac{1}{\hat{s}_i^2}$
$\Rightarrow \mathbf{b}_w=(\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{Y}$ 

Sanity Check: $MSE_w\approx 1$ (since $MSE_w\approx \hat{k}$)

![[WLR procedure.png]]
Inference in WLS case:
- $S^2(\mathbf{b}_w)=MSE_w(\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\rightarrow$ CI's, PI's etc.
- $Var(\mathbf{b}_w)=1(\mathbf{X}^T\mathbf{WX})^{-1}$ 

### Ridge Regression

Idea: reduce variance, but sacrifice unbiasedness
$$
\begin{array}{rl}
X_{ik},Y_i &\rightarrow \text{correlation transform(standardized)}\\[8pt]
& \Rightarrow OLS \quad \mathbf{b}=\mathbf{r}_{XX}^{-1}\mathbf{r}_{YX}\\[8pt]
& \Rightarrow \text{Ridge Regression} \quad \mathbf{b}^R=(\mathbf{r}_{XX}+c\mathbf{I})^{-1}\mathbf{r}_{YX}
\end{array}
$$
where $\mathbf{I}$ is $p-1$ by $p-1$ identity matrix, $c$ is an appropriately chosen constant.

#### P.O.V. 1: Minimization w/ regularization

Consider
$$
\mathbf{b}_R=\arg\min_{\boldsymbol{\beta}\in\mathbb{R}^p}(\|\mathbf{Y}-\mathbf{X}\boldsymbol{\beta}\|^2+c\|\boldsymbol{\beta}\|^2)
$$
where $c\|\boldsymbol{\beta}\|^2$ is regularization term for penalizing large $\beta$'s
- If $c=0$, $\mathbf{b}_R=\mathbf{b}^{OLS}$
- If $c\rightarrow\infty$, $\mathbf{b}_R\rightarrow 0$
- As $c\rightarrow \infty$, the bias $\uparrow$

#### P.O.V. 2: Minimization w/ Constraints

Consider
$$
\begin{array}{rl}
\mathbf{b}=\arg\min_{\boldsymbol{\beta}}\|\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\|^2\\[8pt]

\text{subject to }\|\boldsymbol{\beta}\|^2\leq t
\end{array}
$$
where $t$ is constant.

#### P.O.V. 3: Linear Algebra

$$
\begin{array}{rl}
\text{Multicollinearity} \Rightarrow \det(\mathbf{X}^T\mathbf{X})&\simeq0\\
(\text{eigenvals of } \mathbf{X}^T\mathbf{X})&=\lambda_1\cdot\lambda_2\cdots\lambda_p\\
\text{eigenvals }(\mathbf{X}^T\mathbf{X}+c\mathbf{I})\\
\det(\mathbf{X}^T\mathbf{X}+c\mathbf{I})
\end{array}
$$

#### How's c chosen?

- Plot $\mathbf{b}_R$ for several $c\in[0,1]$ $\Leftarrow$ Trace Plot
- Plot $VIF_k$ (variance inflation factor for $k$-th pred.) for several $c\in[0,1]$
- Good Choice of $c$: estimates stop changing rapidly ("elbow")
![[choose c for ridge regression.jpg]]

Downside:
- subjective choice of $c$
- no distributional results(no CI's, no PIs etc.)
- Solution: Bootstrap

### Robust Regression

OLS is **sensitive** to outlying observations(w.r.t $X$ or $Y$ or both), Robust Regression dampens influential observations.

#### Robust Regression 1

$$
\begin{cases}
LAR=\text{least absolute residuals}\\
LAD=\text{least absolute deviations}\\
L^1=L^1 \text{ norm}
\end{cases}
$$
Objective function:
$$
\min \sum_{i=1}^n\vert Y_i-(\beta_0+\beta_1X_{i,1}+\cdots+\beta_{p-1}X_{i,p-1}) \vert
$$
We can find that $u^2$ has larger values than $|u|$ for $|u|>1$, i.e. larger penalty.

- upside: dampens influential observations
- downside: non-unique solutions, no closed form solutions(no derivative)

#### Robust Regression 2: LMS = least median of squares

Consider
$$
\min_{\boldsymbol{\beta}} \operatorname{median}\,[Y_i-(\beta_0+\beta_1X_{i,1}+\cdots+\beta_{p-1}X_{i,p-1})]^2\Rightarrow \operatorname{median}\,[e_i]^2
$$

Median is more robust than mean.

#### Robust Regression 3: IRLS = Iteratively Reweighted Least Squares

Similar to WLS, but influential pts are downweighted, but non-influential pts are the same
$$
\min\sum_iw_i(e_i)e_i^2\rightarrow \min\sum_iw_i(u_i)u_i^2
$$
where $u_i=\frac{e_i}{MAD}$, $MAD$ is median absolute deviation, $MAD=\frac{1}{0.6745}\operatorname{median}(e_i-\operatorname{median}(e_i))$, $w_i(u_i)$ is function of residuals, tells us how to down weight or not.

**Huber:**
$$
w_i(u)=\begin{cases}
1,|u|<k\\
\frac{k}{|u|},|u|>k
\end{cases}
$$
Generally, $k=1.345$ , $\min\sum_i\frac{k}{|u_i|}u_i^2=\sum_i k|u_i|$ 

**Bisquare:**
$$
w_i(u)=\begin{cases}
(1-(\frac{u}{k}))^2,|u|<k\\
0,|u|>k
\end{cases}
$$
Generally, $k=4.685$. Bisquare function means that if $|u|>k$ then we ignore these observations.

Generally, start with Huber $\Rightarrow$ Switch to Bisquare

#### Properties of Robust Regression

- Regression function must be properly specified
- Robust Regression can identify multiple outliers (low weight $\rightarrow$ disregard)
- Robust Regression is used to confirm the OLS:
	if $b_k^{RR}\approx b_k^{LS}\Rightarrow$ no influential cases

**Downsides:**
- can't obtain CI's, PI's etc (only via bootstrap)
- RR: only deals w/outlying $Y$ observations
- outlying $X$-observations: bounced influence regions 

---