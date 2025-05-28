
$$
\begin{array}{}
\mathbf{Y}=\mathbf{X}\beta+\varepsilon\\
\begin{bmatrix}Y_1\\\vdots\\Y_n\end{bmatrix}=\begin{bmatrix}1&X_{11}&\cdots&X_{1,p-1}\\\vdots&\vdots&&\vdots\\1&X_{n1}&\cdots&X_{n,p-1}\end{bmatrix}\begin{bmatrix}\beta_0\\\vdots\\\beta_{p-1}\end{bmatrix}+\begin{bmatrix}\varepsilon_1\\\vdots\\\varepsilon_n\end{bmatrix}
\end{array}
$$
where  $\mathbf{X}$ is design matrix, $\underline{\varepsilon}\sim N_n(\underline{0},\sigma^2\mathbf{I}_n)$.
- $\beta_0$ is mean response when all predictors $=0$ ;
- $\beta_k$ measures that increase in mean response when $x_k$ is increase by $1$ and other predictors are constant.
- $\beta_0+\beta_1x_{i,1}+\cdots+\beta_{p-1}x_{i,p-1}$ is hyperplane in $\mathbb{R}^p$ 
- In general, we can fit a surface to our responses:
$$
Y_i=c_{i,0}\beta_0+c_{i,1}\beta_1+\cdots+c_{i,p-1}\beta_{p-1}+\varepsilon_i
$$
	where $c_{i,j}$ are any functions of predictor's $X_i$'s or constant.

>[! Example] Categorical predictors with C categories
>We will use "dummy coding": create $C-1$ binary variables. Then we have design matrix as below
>$$
>\begin{bmatrix}
>1&1&0&0\\
>1&0&1&0\\
>1&0&0&1
>\end{bmatrix}
>$$
>>[! Question]
>>Why $C$ variable is not good?
>>
>>**Ans:**
>>Because $\mathbf{X}$ is not full rank, then we have $\mathbf{X}^T\mathbf{X}$ is not invertible.

### Model Fit

#### Method of Least Squares

$$
Q(\beta)=\underset{\beta}{\min}\|\mathbf{Y}-\mathbf{X\beta}\|^2
$$
Then we can get **Normal Equation**
$$
\frac{\partial Q}{\partial \beta}=-2\mathbf{X}^T\mathbf{Y}+2\mathbf{X}^T\mathbf{X}\beta=0\Rightarrow\mathbf{X}^T\mathbf{X}\beta=\mathbf{X}^T\mathbf{Y}
$$
Therefore, we have
$$
\mathbf{b}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}
$$
>[!Question]
>When is $\mathbf{X}^T\mathbf{X}$ invertible?
>
>**Ans:**
>
>If $\mathbf{X}$ is full rank (not rank deficient) i.e. $\text{rank}(\mathbf{X})=p$. In other words, $\mathbf{X}$ has $p$ linear independence columns.

#### Least Squares and MLE

Similar to SLR, we still have MLE and least squares equivalent.

#### Gauss-Markov Theorem for high dimensions

We first define **Linear Unbiased Estimator Class**

>[!Note] Linear Unbiased Estimator Class
>**Definition:**
>
>Let $\mathbf{C}\in\mathbb{R}^p$ is a constant vector, then we have
>$$
>u(\mathbf{C}^T\beta):=\{\mathbf{a}^T\mathbf{y}|\mathbf{a}\in\mathbb{R}^p,E[\mathbf{a}^T\mathbf{y}]=\mathbf{C}^T\beta\}
>$$
>We call $u(\mathbf{C}^T\beta)$ a **Linear Unbiased Estimator Class** of $\mathbf{C}^T\beta$ based on $\mathbf{y}$.

We note that $E[\mathbf{C}^T\mathbf{b}]=\mathbf{C}^T\beta$ and $\mathbf{C}^T\mathbf{b}=\mathbf{C}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$, which mean that $\mathbf{C}^T\mathbf{b}\in u(\mathbf{C}^T\beta)$.

Similar to SLR, we have Gauss-Markov Theorem under high dimensions

>[!Note] Gauss-Markov Theorem for High Dimensions
>Under MLR model, estimate $\mathbf{C}^T\mathbf{b}$ is **BLUE** in $u(\mathbf{C}^T\beta)$.
>
>**Proof:**
>
>$\forall \mathbf{a}^T\mathbf{Y}\in u(\mathbf{C}^T\beta)$, $E(\mathbf{a}^T\mathbf{Y})=\mathbf{C\beta}$, then we have $\mathbf{a}^T\mathbf{X\beta}=\mathbf{C^T\beta}$. By the arbitrariness of $\beta$, we have $\mathbf{a}^T\mathbf{X}=\mathbf{C}^T$.
>$$
>\begin{array}{rl}
>Var(\mathbf{a}^T\mathbf{Y})&=E[(\mathbf{a}^T\mathbf{Y}-\mathbf{C}^T\beta)^2]\\
>&=E[(\mathbf{a}^T\mathbf{Y}-\mathbf{C}^T\hat\beta+\mathbf{C}^T\hat\beta-\mathbf{C}^T\beta)^2]\\
>&=E[(\mathbf{a}^T\mathbf{Y}-\mathbf{C}^T\hat\beta)^2]+Var(\mathbf{C}^T\hat\beta)+2E[(\mathbf{a}^T\mathbf{Y}-\mathbf{C}^T\hat\beta)(\mathbf{C}^T\hat\beta-\mathbf{C}^T\beta)]\\
>\end{array}
>$$
>For the first term is greater than or equal to $0$, we only need to consider the third term
>$$
>\begin{array}{rl}
>E[(\mathbf{a}^T\mathbf{Y}-\mathbf{C}^T\hat\beta)(\mathbf{C}^T\hat\beta-\mathbf{C}^T\beta)]&=Cov(\mathbf{a}^T\mathbf{Y}-\mathbf{C}^T\hat\beta,\mathbf{C}^T\hat\beta-\mathbf{C}^T\beta)\\
>&=Cov(\mathbf{a}^T\mathbf{Y}-\mathbf{C}^T\hat\beta,\mathbf{C}^T\hat\beta)\\
>&=Cov(\mathbf{a}^T\mathbf{Y},\mathbf{C}^T\hat\beta)-Var(\mathbf{C}^T\hat\beta)\\
>&\xlongequal{\mathbf{Y}=\mathbf{X\beta}+\varepsilon} Cov(\mathbf{a}^T\varepsilon,\mathbf{C^T(X^T X)^{-1}X^T}\varepsilon)-Var(\mathbf{C}^T\hat\beta)\\
>&=\sigma^2\mathbf{C^T(X^T X)^{-1}X^T}-\sigma^2\mathbf{C^T(X^T X)^{-1}X^T}\\
>&=0
>\end{array}
>$$
>Therefore, we have $Var(\mathbf{a}^T\mathbf{Y})\geq Var(\mathbf{C^T b})$. The equality holds if and only if $\mathbf{a}^T\mathbf{Y}=\mathbf{C^T b}$.

#### Inference on $\beta_k$'s

##### LS Estimate

$$
E(\mathbf b)=\beta\quad Var(\mathbf{b})=\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}
$$
$$
\Rightarrow s^2(\mathbf{b})=MSE(\mathbf{X}^T\mathbf{X})^{-1}\in\mathbb{R}^p\quad s^2(b_k)=MSE(\mathbf{X}^T\mathbf{X})^{-1}_{k+1,k+1}
$$

##### CI

$$
CI_k:[b_k\pm s(b_k)\cdot t(1-\frac{\alpha}{2},n-p)]
$$

##### HT(t-test)

$$
H_0:\beta_k=0\quad \text{vs}\quad H_a:\beta_k\neq0
$$
We have t-statistic that
$$
t^*=\frac{b_k}{s(b_k)}\thicksim t_{n-p}
$$
Decision Rule:
- $|t^*|\leq t(1-\frac{\alpha}2,n-p)\Rightarrow H_0$
- $|t^*|> t(1-\frac{\alpha}2,n-p)\Rightarrow H_a$

##### Joint Estimation of $\beta_k$

$$
\text{Bonferroni}:[b_k\pm SE(b_k)\cdot t(1-\frac{\alpha}{2g},n-p)]
$$
for $g$ different $\beta_k$'s.

#### ANOVA Table for MLR

|                SS                 |               df               |          MS           |    F-statistic    |
| :-------------------------------: | :----------------------------: | :-------------------: | :---------------: |
| $SSR$(**Regression** of squares)  |   $df_R(p-1,\text{for SLR})$   | $MSR=\frac{SSR}{p-1}$ | $\frac{MSR}{MSE}$ |
|  $SSE$(**Error** sum of squares)  |   $df_E(n-p,\text{for SLR})$   | $MSE=\frac{SSE}{n-p}$ |         -         |
| $SST_o$(**Total** sum of squares) | $df_{T_o}(n-1,\text{for SLR})$ |           -           |         -         |
![[Geometry ANOVA.jpg]]

### Model Diagnostics and Remedial

只要是针对系数显著性的检验，我们都可以使用 General Linear Test，选定全模型和减模型构造 F 统计量即可。
#### Model Utility Test/ F-test/ Overall Fit Test

$$
H_0:\,\beta_1=\beta_2=\cdots=\beta_{p-1}=0\quad \text{vs} \quad H_a:\text{at least one }\beta_k=0
$$
We have F statistics that
$$
F^*=\frac{MSR}{MSE}=\frac{SSR/(p-1)}{SSE/(n-p)}\overset{\text{under }H_0}{\thicksim}F_{p-1,n-p}
$$
Decision Rule:
- $F^*\leq F(1-\alpha,p-1,n-p)\Rightarrow H_0$
- $F^*> F(1-\alpha,p-1,n-p)\Rightarrow H_a$ 

F-test is to test "Is there a useful linear relationship btw $Y$ and $X_1,\cdots,X_{p-1}$".
![[Geometry F-test.jpg]]

>[!Question] CR for F-test
>Why is the confidence region for an F-test is an ellipse/ellipsoid?
>
>**Ans:**
>
>For F-test,
>$$
>F^*\xlongequal{GLT}\frac{(\mathbf{Cb})^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{Cb})}{SSE}\frac{n-p}{q}=\mathbf{b}^T\Omega\mathbf{b}=\text{Quadratic term of }b_i\text{'s}
>$$
>Region: reject region for F-test $F^*\geq \text{C.V.}$; Non-reject region/ CI $\Rightarrow$ $F^*\leq \text{C.V.}=\text{ellipsoid}$.

#### F-test vs t-test

- F-test: test all together ${X_i}$ have a significant linear relationship with $\mathbf{Y}$
- t-test: test individual $\beta_i$ , whether $\beta_i$ should be added to the model when we have other $\beta_k$ 

#### Extra Sum of Squares

Consider Model 1 $Y\thicksim X_1$ and Model 2 $Y\thicksim X_1+X_2$ , we have 

$$
SST_o=SSE(X_1)+SSR(X_1)=SSE(X_1,X_2)+SSR(X_1,X_2)
$$

Notice that for unexplained variance $SSE(X_1)\geq SSE(X_1,X_2)$, for explained variance $SSR(X_1)\leq SSR(X_1,X_2)$ , we define extra sum of squares that

$$
\begin{array}{}
SSR(X_2|X_1)&=&SSE(X_1)-SSE(X_1,X_2)\quad(\text{Reduction in unexplained var})\\
&=&SSR(X_1,X_2)-SSR(X_1)\quad(\text{Increase in explained var})
\end{array}
$$

Extra sum of squares measures how much is being explained by adding $X_2$. 

>[! question] 
>Why $SSE(X_1,X_2)\leq SSE(X_1)$?
>
>**Ans:**
>$$
>SSE(X_1)=\sum_{i=1}^n(Y_i-b_0-b_1X_{i1}-0X_{i2})^2\geq \sum_{i=1}^n(Y_i-b_0'-b_1'X_{i1}-b_2'X_{i2})^2=SSE(X_1,X_2)
>$$

We can write another ANOVA table

| Source of Variation | SS                      | df                 | MS                      |
| ------------------- | ----------------------- | ------------------ | ----------------------- |
| Regression          | $SSR(x_1,x_2,x_3)$      | $df=3$             | $MSR(x_1,x_2,x_3)$      |
| $x_1$               | $SSR(x_1)$              | $df=1$             | $MSR(x_1)$              |
| $x_2\vert x_1$      | $SSR(x_2\vert x_1)$     | $df=(3-1)-(2-1)=1$ | $MSR(x_2\vert x_1)$     |
| $x_3\vert x_1,x_2$  | $SSR(x_3\vert x_1,x_2)$ | $df=(4-1)-(3-1)=1$ | $MSR(x_3\vert x_1,x_2)$ |
| Error               | $SSE(x_1,x_2,x_3)$      | $df=n-4$           | $MSE(x_1,x_2,x_3)$      |
| Total               | $SST_o$                 | $df=n-1$           |                         |
More, we can also calculate
$$
\begin{array}{}
SSR(x_2,x_3|x_1)&=&SSR(x_1,x_2,x_3)-SSR(x_1)\\
&=& df(4-1)-df(2-1)\\
&=& SSR(x_3|x_1,x_2)+SSR(x_2|x_1)\\
&=& df(1)+df(1)
\end{array}
$$
with $df=2$.

We need to test if $SSR(x_2|x_1)$ is large enough. If so, it means that we need to add $x_2$ to our model. We use [[Simple Linear Regression (SLR)#General Linear Test|General Linear Test]] for this
$$
\begin{array}{}
(R): Y_i=\beta_0+\beta_1X_{i,1}+\varepsilon_i\\
(F):Y_i=\beta_0+\beta_1X_{i,1}+\beta_2X_{i,2}+\varepsilon_i
\end{array}
$$
$$
\begin{array}{}
F^*&=&\frac{SSE(R)-SSE(F)}{df_R-df_F}/\frac{SSE(F)}{df_F}\\
&=&\frac{SSE(x_1)-SSE(x_1,x_2)}{(n-2)-(n-3)}/\frac{SSE(x_1,x_2)}{(n-3)}\\
&=&\frac{SSR(x_2|x_1)}{1}/\frac{SSE(x_1,x_2)}{n-3}\\
&=&\frac{MSR(x_2|x_1)}{MSE(x_1,x_2)}\overset{H_0}\thicksim F_{1,n-3}
\end{array}
$$
Decision Rule:
- if $F^*>F(1-\alpha,1,n-3)\Rightarrow$ should add $X_2$ 
	Equivalent to a t-test (marginal test) on individual coefficient $\beta_2$ - $H_0:\beta_2=0$ for the model
$$
\begin{array}{}
Y=\beta_0+\beta_1X_1+\beta_2X_2+\varepsilon\\
[t^*]^2=[\frac{b_2}{s(b_2)}]^2=F^*=\frac{MSR(x_2|x_1)}{MSE(x_1,x_2)}
\end{array}
$$
	Marginal t-test measures that is $X_2$ worth adding given that $X_1$ is already in the model.

##### Geometry

$$
SST_o=SSR(X_1)+SSR(X_2|X_1)+SSE(X_1,X_2)
$$
![[Geometry Extra Sum of Squares.jpg]]

#### Test Any Subset of $\beta_k$'s

e.g. $H_0:\beta_3=\beta_5=0\quad \text{vs}\quad H_a:\beta_3\neq 0 \text{ or }\beta_5\neq 0$ . We can also see subset test as an example of GLT.
$$
H_0: \mathbf{C}\beta=\gamma\quad\text{vs}\quad H_a:\mathbf{C}\beta\neq\gamma
$$
We have F statistic that
$$
\begin{array}{}
F^* &=& \frac{(\mathbf{Cb}-\gamma)^T[Var(\mathbf{Cb})]^{-1}(\mathbf{Cb}-\gamma)/q}{SSE/\sigma^2/(n-p)}\\
&=&\frac{(\mathbf{Cb}-\gamma)^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{Cb}-\gamma)}{SSE}\frac{n-p}{q}
\end{array}
$$
where $q$ is the rank of $\mathbf{C}$. (Most of time $C$ is full rank)

$(\mathbf{Cb}-\mathbf{C\beta})^T[Var(\mathbf{Cb})]^{-1}(\mathbf{Cb}-\mathbf{C\beta})$ measures how far $\mathbf{Cb}$ from $\mathbf{C\beta}$ .

>[!Note] 如何确定矩阵 $\mathbf{C}$
>核心是确定 $H_0$ 的假设中究竟包含了几个等式，等式的数量即为 $\mathbf{C}$ 的行数。一般等式之间都是互不相关，所以 $\mathbf{C}$ 一般是满秩。

![[Geometry subset beta.jpg]]


|          GLT           | Is change in explained variation from<br>Reduced model to Full model enough<br>comparing to unexplained variation in<br>Full model? |
| :--------------------: | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Model Utility Test** | (R):$Y=\beta_0+\varepsilon\rightarrow$ (F)<br>      no $x$ at all   $\rightarrow$  all $x$                                          |
|  **Marginal t-test**   | (R): no $\beta_k\rightarrow$ (F)<br>only missing $x_k$   $\rightarrow$   all $x$                                                    |
|    **Subset Test**     | (R): $\mathbf{C}\beta=\gamma\rightarrow$ (F)<br>some subspace of $col(X)$  $\rightarrow$  all $x$                                   |

#### Coefficient of Partial Determination

$$
R^2_{Y2|1}=\frac{SSE(x_1)-SSE(x_1,x_2)}{SSE(x_1)}=\frac{SSR(x_2|x_1)}{SSE(x_1)}
$$
It measures relative reduction in explained variance in $Y$ when $X_2$ is added to $Y\sim X_1$ model.

Alternatively,
$$
\begin{array}{}
Y\thicksim X_1\Rightarrow e_i(Y|X_1)=Y_i-\hat Y_i(X_1)=\text{part of } Y \text{ unexplained by }X_1\\
X_2\thicksim X_1\Rightarrow e_i(X_2|X_1)=X_{i2}-\hat X_{i2}(X_1)=\text{part of } X_2 \text{ unexplained by }X_1
\end{array}
$$
Regression: $e_i(Y|X_1)\sim e_i(X_2|X_1)$

Then we compute coefficient of simple determination $R^2$, we can find it same as $R^2_{Y2|1}$.

Moreover,
$$
\begin{array}{}
\hat Y_i=b_0+b_1X_1+b_2X_2\\
\hat e_i(Y|X_1)=\tilde{b}_0+b_2e_i(X_2|X_1)
\end{array}
$$
![[Geometry e_i vs e_i.jpg]]

We can plot $e_i(Y|X_1)\quad\text{vs}\quad e_i(X_2|X_1)$ .

#### Coefficient of Multiple Determination $R^2$

$$
R^2=\frac{SSR}{SST_o}=1-\frac{SSE}{SST_o}
$$
$R^2$ measures relative reduction in variation in $Y$ due to all predictors $X_1,\cdots,X_{p-1}$.
![[Geometry R^2.jpg]]

#### Adjusted $R^2$

$$
R^2_{adj}=1-\frac{MSE}{MST_o}=1-\frac{SSE/(n-p)}{SST_o/(n-1)}=1-\frac{n-1}{n-p}\frac{SSE}{SST_o}=1-\frac{s^2(e_i)}{s^2(Y_i)}
$$
We add penalty term $(n-1)/(n-p)$ for high $p$.

Adjusted $R^2$ measures the relative reduction of estimated variance of $Y$ due to $X_1,\cdots,X_{p-1}$.

#### Hidden Extrapolations

即隐形外推，主要指在模型预测时，模型的训练数据并没有覆盖预测数据所在的区域，但模型仍然试图进行预测的现象。这种预测可能会显得合理，但其可靠性往往较差，因为模型并未在该特征值的范围内接受过训练。

Hidden extrapolation can be easily detected in SLR, but in MLR hard.
![[Hidden Extrapolation.jpg]]

Consider $h_{new,new}=\mathbf{X}_h^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}_h$ 
- If $h_{ii(min)}<h_{new,new}<h_{ii(max)}$, then there is no hidden extrapolation
- If $h_{new,new}$ outside the range $\Rightarrow$ hidden extrapolation

#### Multicollinearity

Reason: Design matrix $\mathbf{X}$ is not full rank $\Rightarrow$ $\mathbf{X}^T\mathbf{X}$ is almost not full rank

Multicollinearity happens if predictors are "similar", they are linearly related.
- $b_k$ will have very large variance
- $SE(b_k)$ is very large
- CI's for $\beta_k$ is wide
- likely $0\in CI(\beta_k)$
- $\beta_k$ will not be significant
- $H_0:\beta_k=0$ is accepted (Marginal t-test)

A few signs of multicollinearity
- individual t-test is not significant, overall F-test is significant
- large SD's for predictors
- non-significant important pred's or opposite relationship

In general, multicollinearity occurs when $corr(X_j,\sum_{i\neq j}a_iX_i)$ is high
$\Rightarrow$ $\det(\mathbf{X}^T\mathbf{X})\approx 0$ $\Rightarrow$ large value of $(\mathbf{X}^T\mathbf{X})^{-1}$ $\Rightarrow$ large SD's for $b_k$ $\Rightarrow$ non significant $\beta_k$'s

>[!Question] 
>Why is $r_{ij}$ (Sample correlations between $X_i,X_j$) not enough to detect multicollinearity?
>
>**Ans:**
>
>- Multicollinearity may involve combinations of more than two variables.
>- The correlation coefficient can only detect cases of perfect linear correlation
>- The correlation coefficient ignores multiple correlations and variance inflation
>- There may be hidden correlations. Even if two variables are not highly correlated, their combination may still be highly correlated with another variable.

For model utility test
- $H_a:\beta_1\neq0 \text{ or } \beta_2\neq 0$ is accepted (F-test)
- $X_2$ should not be added to $Y\sim X_1$

Then we get an incompatible result!
![[CR vs CI multicollinearity.jpg]]
##### Variance Inflation Factor(Var of $b_k$)

After correlation transform: $Var(\mathbf{b}^*)=(\sigma^*)^2\mathbf{r}_{xx}^{-1}$ .

We define
$$
VIF_k=[\mathbf{r}_{xx}^{-1}]_{k,k}=\text{how much is }(\sigma^*)^2\text{ inflated by to get }Var(b_k^*)
$$
And $Var(b_k^*)=(\sigma^*)^2\times VIF_k$. We can also have
$$
VIF_k=\frac{1}{1-R^2_k}
$$
where $R_k^2$ is a coefficient of mult. determs when $X_k$ is regressed on $X_1,\cdots,X_{k-1},X_{k+1},\cdots,X_{p-1}$.
![[VIF.jpg]]

#### Outliers/ Outlying Observations

There are $3$ kinds of outliers
- Outliers with respect to $Y$ (1)
- Outliers with respect to $X$ (2)
- Outliers with respect to $X$ & $Y$ (3)

For MLR, we have
$$
\mathbf{e}=\begin{bmatrix}e_1\\\vdots\\e_n\end{bmatrix}=\mathbf{Y}-\hat {\mathbf{Y}}=(\mathbf{I}_n-\mathbf{H})\mathbf{Y}\Rightarrow Var(\mathbf{e})=\sigma^2(\mathbf{I}_n-\mathbf{H})
$$
So we have
$$
Var(e_i)=\sigma^2(1-h_{ii})
$$
which can be seen in [[Simple Linear Regression (SLR)#Vector of residuals e|variance of residuals]]. We can also prove that
$$
h_{ii}=[\mathbf{H}]_{ii}=[\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T]_{ii}=\mathbf{x}_i(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}_i^T
$$
##### Outliers with respect to Y

###### Internally Studentized Residuals(ISR)

$$
ISR=r_i=\frac{e_i}{S(e_i)}=\frac{e_i}{\sqrt{\hat\sigma^2(1-h_{ii})}}\quad(\text{correct studentized})
$$
semi-studentized: $e_i^*=\frac{e_i}{\sqrt{MSE}}$ 

We can detect like (1) and (3) by ISR plot. 
- $|r_i|>2\text{ or }3\Rightarrow$ potential outliers
- For smaller samples, 3 may be a more commonly used standard; for larger samples, 2 is a more common threshold.

###### Deleted Residuals

$$
d_i=Y_i-\hat Y_{i(i)}=\frac{e_i}{1-h_{ii}}
$$
where $\hat Y_{i(i)}$ is estimate of $Y_i$ by the model without ith observation $(\mathbf{X}_i,Y_i)$.

The fraction above means we don't need to refit $n$ different regression's for each observations.

We can also calculate
$$
s^2(d_i)=\hat{Var(d_i)}=MSE_{(i)}[1-\mathbf{X}^T_i(\mathbf{X_{(i)}}^T\mathbf{X_{(i)}})^{-1}\mathbf{X_{i}}]=\frac{MSE_{(i)}}{1-h_{ii}}
$$
where $MSE_{(i)}=$ MSE of fitted model without case $i$
$$
SSE=(n-p-1)MSE_{(i)}+\frac{e_i^2}{1-h_{ii}}
$$

###### Externally Studentized Residuals(ESR)

$$
t_i=\frac{d_i}{s(d_i)}=e_i\times\sqrt{\frac{n-p-1}{SSE(1-h_{ii})-e_i^2}}\thicksim t_{n-p-1}
$$
Test for outliers: ($n$ tests on each observations)

ith observation is outlier if
$$
|t_i|>t(1-\frac{\alpha}{2n},n-p-1)
$$

|     |        Residuals         |                       Studentized                       |            Diagnostics             |
| :-: | :----------------------: | :-----------------------------------------------------: | :--------------------------------: |
| ISR |    $e_i=Y_i-\hat Y_i$    |         $r_i=\frac{e_i}{\sqrt{SSE(1-h_{ii})}}$          |      detect outliers wrt $Y$       |
| ESR | $d_i=Y_i-\hat{Y}_{i(i)}$ | $t_i=e_i\times\sqrt{\frac{n-p-1}{SSE(1-h_{ii})-e_i^2}}$ | **better** detect outliers wrt $Y$ |

##### Outliers with Respect to X

$$
\begin{array}{}
\mathbf{H}_{ii}&=&\text{``distance measure in a predictor space"}\\
&\approx&\text{difference from }\bar{\mathbf{X}}=(\bar{\mathbf{X}}_1,\cdots,\bar{\mathbf{X}}_{p-1})\\
&=&\text{distance to the center of the data}\\
&=&\text{influence of }X_i\text{ on fitted values}\\
\hat Y_i&=&(\mathbf{HY})_i=h_{i1}Y_1+\cdots+h_{ii}Y_i+\cdots+h_{in}Y_n
\end{array}
$$
Largest possible $h_{ii}=1$ $\Rightarrow$ $Var(e_i)=\sigma^2(1-h_{ii})=0\Rightarrow Y_i=\hat Y_i$ .

Rule of Thumb for Leverage: Alternative Rule of Thumb
- $h_{ii}>0.5=\text{``high leverage"}$
- $0.2<h_{ii}\leq 0.5=\text{``moderate leverage"}$
- Alternatively, if $h_{ii}>2\times(\text{coverage leverage})=2\frac{p}{n}$ , high leverage
	![[leverage.jpg]]

##### Measures of Influence

###### DFFITS(删失拟合值差异/拟合差异诊断)

It measures how sensitive the fit when one case is removed. We define
$$
\begin{array}{}
(DFFITS)_i&=&\frac{\hat Y_i-\hat Y_{i(i)}}{\sqrt{MSE_{(i)}h_{ii}}}\\
&=&e_i\times\sqrt{\frac{n-p-1}{SSE(1-h_{ii})-e_i^2}}\times(\frac{h_{ii}}{1-h_{ii}})^{\frac{1}{2}}\\
&=&t_i\times(\frac{h_{ii}}{1-h_{ii}})^{\frac{1}{2}}
\end{array}
$$
![[DFFITS.jpg]]

Which points are influential?
- outlier wrt $Y$ & moderate leverage $\Rightarrow$ $(DFFITS)_i$ high
- outlier wrt $X$ (high $h_{ii}$) & moderate leverage $\Rightarrow$ $(DFFITS)_i$ high
- outlier wrt $X$ & $Y$ (high $t_i$ and high $h_{ii}$) $\Rightarrow$ $(DFFITS)_i$ high $\Rightarrow$ Influential

Rule of Thumb:
- $|(DFFITS)_i|>1\Rightarrow$ influential obs'n on $\hat Y_i$ (for small or medium datasets)
- $|(DFFITS)_i|>2\sqrt{\frac{p}{n}}\Rightarrow$ influential obs'n on $\hat Y_i$ (for large datasets)

###### Cook's Distance

Cook's distance ($D_i$) measures the influence of the ith case $\mathbf{X}_i$ on all fitted values.
$$
D_i=\frac{\sum_{j=1}^n(\hat Y_j-\hat Y_{j(i)})^2}{p\times MSE}=\frac{\|\mathbf{\hat Y}-\mathbf{\hat Y_{(i)}}\|^2}{p\times MSE}=\frac{e_i^2}{p\cdot MSE}\frac{h_{ii}}{(1-h_{ii})^2}=D(e_i,h_{ii})
$$
![[Cook distance.jpg]]
![[cook distance 1.jpg]]

###### $DFBETAS_{k(i)}$(删失回归系数差异/回归系数差异诊断)

It measures that a change in the kth estimated coefficient $b_k$, when observation $\mathbf{X}_i$ is removed, studentized.
$$
DFBETAS_{k(i)}=\frac{b_k-b_{k(i)}}{\sqrt{MSE_{(i)}C_{kk}}}
$$
where $k=0,\cdots,p-1,\, i=1,2,\cdots,n$ and $C_{kk}=[(\mathbf{X}^T\mathbf{X})^{-1}]_{k+1,k+1}$.

Rule of Thumb:
- $|DFBETAS_{k(i)}|>1$ influential for small/medium datasets
- $|DFBETAS_{k(i)}|>\frac{2}{\sqrt{n}}$ for large datasets

Note:$(\text{Cook's dist})_i=D_i\approx \sum_{k=0}^{p-1}[(DFBETAS)_{k(i)}]$ for each fixed $i$.

![[MSE comparison.jpg]]

#### Diagnostics/ Remedial Measures

##### Plot

- $e_i\,\text{vs}\, \hat Y_i$: non-constancy of variance; non-linear regression 
- Histogram & QQ plot of $e_i$'s: non-normal
- $e_i\,\text{vs}\,X_{i,k}$ for all $k=1,\cdots,p-1$: non-constancy of variance; non-linear regression
- $e_i\,\text{vs}\,\text{omitted predictors}$: missing predictors
- $e_i\,\text{or}\,e_i^2\,\text{vs}\,\hat Y_i$: non-constancy of variance
- $e_i\,\text{vs}\,(X_{i,k},X_{i,j})$: missing interaction term
	- systematic pattern (like quadratic trend/ other curve trend), then interaction effect exists
	- no systematic pattern, then no interaction effect exists
- Scatterplot matrix: non-linear regression, multicollinearity
- Rare: 3D scatterplot of $Y_i\,\text{vs}\,(X_{i,k},X_{i,j})$: non-linear regression, multicollinearity, interaction term
- Added variable plot/ partial regression plot
	consider
	- $Y\sim \text{all pred's except }X_k\Rightarrow e_i(Y)$, which is unexplained variation of $Y$ by $x_1,\cdots,x_{k-1},x_{k+1},\cdots,x_n$ 
	- $X_k\sim \text{all pred's except }X_k\Rightarrow e_i(X_k)$, which is unexplained variation of $X_k$ by $x_1,\cdots,x_{k-1},x_{k+1},\cdots,x_n$
	Plot $e(Y_i)v.s.e_i(X_k)$
	![[added variable plot.jpg]]

##### Tests

- Shapiro-Wilk Test: normal
- Brown-Frosythe Test: constancy of Error Variance
- Correlation Matrix

##### Simple Remedial Measures

- If variance is non-constant $\Rightarrow$ Transform $\mathbf{Y}$ or transform $\mathbf{X}$ & $\mathbf{Y}$ 
- If relationship is nonlinear $\Rightarrow$ Transform $X$ or add a new nonlinear predictor $f(x)$ 

### Model Prediction

#### Estimation of the Mean Response at $\mathbf{X}_h$

$$
\hat Y_h=\mathbf{X}^T_h\mathbf{b}=b_0+b_1x_{h,1}+\cdots+b_{p-1}x_{h,p-1}
$$
$$
Var(\hat Y_h)=Var(\mathbf{X}^T_h\mathbf{b})=\sigma^2\mathbf{X}^T_h(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}_h
$$

#### CI for Mean Response

$$
[\hat Y_h\pm s(\hat Y_h)\cdot t(1-\frac{\alpha}{2},n-p)]
$$
where $s(\hat Y_h)=\sqrt{MSE\cdot \mathbf{X}^T_h(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}_h}$ (scalar, 标量).

#### CI for New Prediction

$$
\hat Y_h\pm t(1-\frac{\alpha}2,n-p)SE(\text{pred})
$$
where $SE^2(\text{pred})=MSE(1+\mathbf{X}^T_h(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}_h)$.

#### CI of Prediction for $g$ different values of $X_h$'s

$$
\text{Bonferroni}: \hat Y_h\pm SE(\hat Y_h)t(1-\frac{\alpha}{2g},n-p)
$$
Scheffe 会更保守 (not cover)，相比 Bonferroni 会更宽。
#### Working Hoteling Confidence Region

We can get the confidence region for the whole regression surface (for all values of $X_i$'s)
$$
\hat Y_h\pm W\cdot SE(\hat Y_h)
$$
where $W=\sqrt{pF(1-\alpha,p,n-p)}$. ($p$ instead of $2$)

### Polynomial Regression

$$
Y_i=\beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\beta_{12}X_{i1}X_{i2}+\beta_{11}X_{i1}^2+\beta_{22}X_{i2}^2+\varepsilon_i
$$
where $X_{i1}X_{i2}$ is interaction term.

#### Predictors: Qualitative/ Categorical

- If model does not have interaction term, then we can get two regression lines with same slope under qualitative predictor;
- If model has interaction term, then we get two regression lines with different slops under qualitative predictor.

Brown-Frosythe Test for constancy of Error Variance
Breusch-Pagan Test for constancy of Error Variance

---