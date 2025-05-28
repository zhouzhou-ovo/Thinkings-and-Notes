
$$
Y_i=\beta_0+\beta_1X_i+\varepsilon_i
$$
# What does "linear" mean?

Linear means that the relationship of **unknown parameters** $\beta_0,\beta_1$ is linear.

# Gauss-Markov Hypothesis

For the random noise term $\varepsilon_i$ , we have
- $E\,{\varepsilon_i}=0$
- $Var\,\varepsilon_i = \sigma^2$ 
- $cov(\varepsilon_i,\varepsilon_j)=0,\quad i\neq j$

# Model Fit

## Method of Least Squares

Let $Q(\beta_0,\beta_1):= \sum_{i=1}^n(Y_i-\beta_0-\beta_1X_i)^2$ , our goal is to get $b_0,b_1$, such that
$$
(b_0,b_1)=\underset{\beta_0,\beta_1}{\arg\min}\,Q(\beta_0,\beta_1)
$$
Consider partial derivative of $Q(\beta_0,\beta_1)$ , we can get
$$
\begin{cases}
\frac{\partial Q}{\partial \beta_0}|_{\beta_0=b_0}=\sum_{i=1}^n2(Y_i-\beta_0-\beta_1X_i)(-1)=0\\
\frac{\partial Q}{\partial \beta_1}|_{\beta_1=b_1}=\sum_{i=1}^n2(Y_i-\beta_0-\beta_1X_i)(-X_i)=0
\end{cases}
$$
Then, we have
$$
\begin{cases}
b_0=\bar Y-b_1\bar X\\
b_1=\frac{S_{xy}}{S_{xx}}
\end{cases}
$$
## MLE = LS in SLR

It is easy that we can get the log-likelihood function, and we can prove that
$$
\begin{array}{}
\max \text{log-likelihood function} \Leftrightarrow \min Q(\beta_0,\beta_1)\\
\Rightarrow
\begin{cases}
\hat{\beta_0}_{MLE}=\hat{\beta_0}_{LS}=b_0\\
\hat{\beta_1}_{MLE}=\hat{\beta_1}_{LS}=b_1\\
\end{cases}
\end{array}
$$
under normal distribution assumption.

## Regression Coefficient and Residual

### $b_1$ 
#### Distribution of $b_1$
$$
b_1\thicksim N(\beta_1,\frac{\sigma^2}{S_{xx}})
$$

>[!question] How can we make $Var(b_1)$ smaller?
>**Ans:**
> - Decrease $\sigma^2$
> - Increase $n$
> - Make $x_i$ spread more

#### Hypothesis testing of $\beta_1$
$$
H_0:\,\beta_1=0\quad \text{vs}\quad H_a:\beta_1\neq 0
$$
- Meaning of $H_0$: there is no linear relationship between $X_i,Y_i$
- Test Statistic: 
	$$\begin{cases}Z^*=\frac{b_1-\beta_1}{\sqrt{\sigma^2/S_{xx}}}\overset{H_0}{\thicksim} N(0,1)\\
	t^*=\frac{b_1-\beta_1}{\sqrt{s^2/S_{xx}}}=\frac{b_1-\beta_1}{\sqrt{MSE/S_{xx}}}\overset{H_0}\thicksim t_{n-2}\end{cases}$$

#### Confidence Interval for $\beta_1$

$(1-\alpha)100\%$ confidence interval for $\beta_1$
$$
b_1\pm t(1-\alpha/2,n-2)SE(b_1)
$$
where $SE(b_1)=s(b_1)=\sqrt{MSE/S_{xx}}$ .

>[!question]
>What is the connection to the Hypothesis testing?
>
>**Ans:**
>
>reject $H_0$ at level $\alpha$ $\Leftrightarrow$ $0\notin CI_{(1-\alpha)100\%}$ 
>
>"accept" $H_0$ at level $\alpha$ $\Leftrightarrow$ $0\in CI_{(1-\alpha)100\%}$ 

### $b_0$

We can calculate that
$$
\begin{array}{}
b_0\thicksim N(\beta_0,(\frac{1}{n}+\frac{\bar X^2}{S_{xx}})\sigma^2)\\
SE(b_0)=\hat \sigma\sqrt{\frac{1}{n}+\frac{\bar X^2}{S_{xx}}}=\sqrt{MSE}\sqrt{\frac{1}{n}+\frac{\bar X^2}{S_{xx}}}
\end{array}
$$

### $(b_0, b_1)$
#### Joint estimation of $(\beta_0,\beta_1)$

**Idea**: Find a region $R\subseteq \mathbb{R}^2$ s.t.
$$
\Pr((\beta_0,\beta_1)\in R)=1-\alpha
$$

#### Confidence Region of $(\beta_0,\beta_1)$

- If we use $CI_{1-\alpha}$ for $\beta_0,\beta_1$ to calculate confidence region, we will find
	$$
	\Pr[(\beta_0,\beta_1)\in CI_{\beta_0}\times CI_{\beta_1}]\geq1-2\alpha
	$$
	We consider the worth case, $\Pr$ above is $1-2\alpha$, then
	$$
	\Pr[(\beta_0,\beta_1)\in CI^0_{95\%}\times CI^1_{95\%}]\geq0.9\neq0.95
	$$
	So in order get $(1-\alpha)100\%$ CR, we need to consider
	$$
	CR_{1-\alpha}=\begin{cases}b_0\pm t(1-\alpha/4,n-2)SE(b_0)\\
	b_1\pm t(1-\alpha/4,n-2)SE(b_1)
	\end{cases}
	$$

### Gauss-Markov Theorem

>[!note] Theorem(Gauss-Markov)
>Under the SLR model $b_0$ and $b_1$ are **BLUE(MVUE)** = best **linear** **unbiased**
>estimators (**minimum variance** unbiased estimators) 
>
>**Note:** The theorem also holds for [[Multiple Linear Regression(MLR)#Gauss-Markov Theorem for high dimensions|high dimensional situations]].

For the proof of the theorem, we need to prove $3$ parts: linear, unbiased and minimum variance.
- Linear: trivial
- Unbiased: trivial
- Minimum variance:
	we have $2$ methods to prove:
	- Using "**Inserting Item**" trick to prove
	- Using "**Lagrange Multiplier Approach**" to prove

#### Some important Inferences in  proof of Gauss-Markov Thm

For all linear and unbiased estimator $\hat{\beta_1}=\sum_{i=1}^nc_iY_i$ in SLR model, we have
- $\sum_{i=1}^nc_i=0$
- $\sum_{i=1}^nc_ix_i=1$

### Properties of residuals $e_i$

We define $e_i=\hat{\varepsilon_i}=Y_i-\hat{Y_i}$ , then
- $\sum_{i=1}^ne_i=0$
- $\sum_{i=1}^ne^2_i$ is at minimum
- $\sum_{i=1}^nY_i=\sum_{i=1}^n\hat{Y_i}\Leftrightarrow \bar{Y}=\bar{\hat{Y}}$ 
- $\sum_{i=1}^nX_ie_i=0$
- $\sum_{i=1}^n\hat{Y_i}e_i=0$
- Fitted regression line always passes through $(\bar X,\bar Y)$ 

### Different Estimations of $\sigma^2$

- For $Y_1,\cdots,Y_n\overset{iid}{\thicksim}N(\mu,\sigma^2)$ , we have
	$$
	\hat{\sigma^2}=s^2=\frac{1}{n-1}\sum_{i=1}^n(Y_i-\bar Y)^2
	$$
	with degree of freedom $n-1$.
- For SLR model, we have
	$$
	\hat{\sigma^2}=\frac{1}{n-2}\sum_{i=1}^ne_i^2=\frac{1}{n-2}\sum_{i=1}^n(Y_i-\hat{Y_i})^2=\frac{SSE}{n-2}=MSE
	$$
	with degree of freedom $n-2$.

For the proof and discussion of DF, see Discussion of TA.

### ANOVA Analysis

![[deviation of slr.jpg]]

|                SS                 |               df               |          MS           |               E.MS.               |
| :-------------------------------: | :----------------------------: | :-------------------: | :-------------------------------: |
| $SSR$(**Regression** of squares)  |    $df_R(1,\text{for SLR})$    |  $MSR=\frac{SSR}{1}$  | $E(MSR)=\sigma^2+\beta_1^2S_{xx}$ |
|  $SSE$(**Error** sum of squares)  |   $df_E(n-2,\text{for SLR})$   | $MSE=\frac{SSE}{n-2}$ |         $E(MSE)=\sigma^2$         |
| $SST_o$(**Total** sum of squares) | $df_{T_o}(n-1,\text{for SLR})$ |           -           |                 -                 |
- $SST_o=SSE+SSR$
	- $SST_o=\sum(Y_i-\bar Y)^2$ : Total variation in $Y_i$ around the mean
	- $SSE=\sum(Y_i-\hat Y_i)^2$ : Variation in $Y_i$ left after "accounting for $X$", or unexplained variation
	- $SSR=\sum(\hat Y_i-\bar Y)^2$ : Variation in "$Y_i$ due to $X$", or explained variation
	- We want $SSR$ high, $SSE$ low.
- $MSE=\hat\sigma^2=\text{estimator of the error variance}$
- $MSR=\hat\sigma^2(\text{due to }X)=\text{estimator of variance of }Y\text{ due to }X$ 
- If $\beta_1=0\Rightarrow E(MSR)=E(MSE)$ 

## Model Diagnostics and Remedial

### Model Utility test

We can use [[Simple Linear Regression (SLR)#Hypothesis testing of $ beta_1$|t-test]]; we can also use F-test. In fact, the two tests are equivalent.
$$
H_0:\,\beta_1=0\quad \text{vs}\quad H_a:\beta_1\neq 0
$$
- Fact: $\frac{SSE}{\sigma^2}\thicksim\chi^2_{n-2},\quad\frac{SSR}{\sigma^2}\thicksim\chi^2_1(\text{under }H_0),\quad SSR\perp SSE$ 
- Test statistics:
	$$
	\begin{array}{}
	F^*&=&\frac{(SSR/\sigma^2)/1}{(SSE/\sigma^2)/(n-2)}\\&=&\frac{MSR}{MSE}\\
	&=&\frac{\text{estimated explained error}}{\text{estimated unexplained error}}\\
	\end{array}
	$$
- Equivalence to t-test: 
	- $(t^*)^2=F^*$
	- $|t^*|>t(1-\alpha/2,n-2)\Leftrightarrow F(1-\alpha,1,n-2)$
- Comparison with t-test:
	- F-test is always **one-sided**;
	- t-test is more **flexible**, can be **one-sided or two-sided**
	- Two tests can not be used interchangeably.

### General Linear Test

$$
\begin{array}{}
\text{Full Model}:Y_i=\beta_0+\beta_1X_i+\varepsilon_i\\
\text{Reduced Model}:Y_i=\beta_0+\varepsilon_i\\
SSE(F)=\sum(Y_i-\hat Y_i^F)^2=\sum e_i^2=SSE\\
SSE(R)=\sum(Y_i-\hat Y_i^R)^2=\sum(Y_i-b_0^R)^2=\sum(Y_i-\bar Y)^2=SST_o
\end{array}
$$
- $SSE(F)\leq SSE(R)$
- $SSE(F)<<SSE(R)$ , then Full Model is best;
- $SSE(F)\approx SSE(R)$ , then Full Model and Reduced Model are similar.
- Test statistics:
	$$
	\begin{array}{}
	F^*&=&\frac{\text{``ave reduction in unexplained var'n when X is added"}}{\text{``ave unexplained var"}}\\
	&=&\frac{SSE(R)-SSE(F)}{df_R-df_F}/\frac{SSE(F)}{df_F}\thicksim F_{df_R-df_F,df_F}
	\end{array}
	$$
	For our case, we have $F^*=\frac{MSR}{MSE}$ .
- Model Utility Test is a special case of General Linear Test.

### Coefficient of (simple) determination $R^2$

>[!note] Definition:
>$$
>\begin{array}{}
>R^2 &=& \frac{SSR}{SST_o}\\&=&1-\frac{SSE}{SST_o}\\
>&=& \text{a portion variation of }Y\text{ due to }X\\
>&=& \text{a portion of explained variation}
>\end{array}
>$$
- $R^2$ can only be used for **linear model**.

- **Misconceptions regarding $R^2$**
	- High $R^2$ does not mean good fit;
	- High $R^2$ does not mean the model can make good predictions;
	- Low $R^2$ does not mean $X$ and $Y$ are not related.
	All above misconceptions can use quadratic model to explain.

- Coefficient of correlation
	$$
	\begin{array}{}
	\rho(x,y)=corr(x,y)=\pm\sqrt{R^2}\\
	-1\leq \rho(x,y)\leq 1\quad(\text{Using Cauchy-Schwarz Inequation to prove})
	\end{array}
	$$

### Diagnostics Measures

See Lecture 7.

### Remedial Measures

#### Variance Stabilizing Transform

**Idea**: Transform $Y_i$ , such that
$$
Y_i'=h(Y_i),\quad Var(Y_i')\approx const
$$
$$
h(Y)=\int \frac{1}{SD(Y)}dy
$$
Proof: Using Taylor Expansion and linear approximation in Taylor series.

##### Transformation

| Relationship between $\sigma^2,E(Y)$ | Transformation          |
| ------------------------------------ | ----------------------- |
| $\sigma^2\propto E(y)$               | $h(y)=\sqrt y$          |
| $\sigma^2\propto E(y)[1-E(y)]$       | $h(y)=\arcsin \sqrt y$  |
| $\sigma^2\propto [E(y)]^2$           | $h(y)=\log y$           |
| $\sigma^2\propto [E(y)]^3$           | $h(y)=y^{-\frac{1}{2}}$ |
| $\sigma^2\propto [E(y)]^4$           | $h(y)=y^{-1}$           |

>[!question] When and how to transform?
>- Relationship is non-linear and variance unstable $\Rightarrow$ transform $Y$
>- Relationship is non-linear and variance stable $\Rightarrow$ transform $X$
>- Transform $X$ and $Y$
>	- Preserve linear relationship
>	- maintain constant variance

##### Box-Cox Transforms

>[!question] When to use?
>
>**Ans:**
>
>Hard to identity which transformation of $Y$ to use or relationship between $\sigma^2$ and $E(Y)$ .

Box-Cox transforms is a family of  transformations, parametrized by $\lambda$ :
$$
Y'=\begin{cases}
\frac{Y^{\lambda}-1}{\lambda},\quad \lambda\neq0\\
\log Y,\quad \text{for }\lambda=0
\end{cases}
$$

###### Fun Fact

- $\lim_{\lambda\rightarrow 0}\frac{Y^{\lambda}-1}{\lambda}=\log Y$ 
- $\lambda=1,\quad Y'\approx Y$
- $\lambda=2,\quad Y'\approx Y^2$
- $\lambda=-1,\quad Y'\approx \frac{1}Y$
- $\lambda=\frac{1}2,\quad Y'\approx \sqrt Y$

###### Comments

- $\lambda$ is chosen from $\{\pm\frac{n}{2}\}\,n\in \mathbb{N}$
- Defined for $Y>0$ , sometimes need to shift.
- "Best" $\lambda$ is chosen via an MLE approach.
- Need to consider non-linearity and heteroskedasticity going together. Replot diagnostics plots after transforms and check again.

## Model Prediction

Consider $X_h$ , a new level of predictor $X$ , then we have
$$
\begin{array}{}
\hat{Y_h}=b_0+b_1X_h\\
E(Y_h)=\beta_0+\beta_1X_h\\
\widehat{E(Y_h)}=b_0+b_1X_h=\hat{Y_h}
\\E(\hat{Y_h})=\beta_0+\beta_1X_h=E(Y_h)\Rightarrow \hat{Y_h}\text{ is an unbiased estimator of }E(Y_h)\\
Var(\hat{Y_h})=\sigma^2[\frac{1}{n}+\frac{(X_h-\bar X)^2}{S_{xx}}]\\
\hat{Y_h}\thicksim N(EY_h,\sigma^2[\frac{1}{n}+\frac{(X_h-\bar X)^2}{S_{xx}}])
\end{array}
$$
### $(1-\alpha)100\%$ confidence interval for  the mean response $EY_h$

Using statistics
$$
\frac{\hat{Y_h}-EY_h}{\sqrt{SE(\hat{Y_h})}}\thicksim t_{n-2}
$$
$$
\hat{Y_h}\pm t(1-\alpha/2,n-2)\times SE(\hat{Y_h})
$$
where $SE(\hat{Y_h})=\sqrt{MSE(\frac{1}{n}+\frac{(X_h-\bar X)^2}{S_{xx}})}$ .

### $(1-\alpha)100\%$ prediction interval for $Y_h$ at $X_h$

Using statistics
$$
\frac{Y_h-\hat{Y_h}}{SE(prediction)}\thicksim t_{n-2}
$$
$$
\hat{Y_h}\pm t(1-\alpha/2,n-2)\times SE(prediction)
$$
where $SE(prediction)=\sqrt{MSE(1+\frac{1}{n}+\frac{(X_h-\bar X)^2}{S_{xx}})}$ .
- When we calculate $Var(Y_h-\hat{Y_h})$ , we have
	$$cov(Y_h,\hat{Y_h})=cov(\beta_0+\beta_1X_h+\varepsilon_h,b_0+b_1X_h)=0$$
	This is because	$$\varepsilon_h\perp\!\!\!\perp \varepsilon_1,\cdots,\varepsilon_n\rightarrow Y_1,\cdots,Y_n\rightarrow b_0,b_1$$

|           |                        CI                         |                      PI                      |
| --------- | :-----------------------------------------------: | :------------------------------------------: |
| Different | estimating value of the mean response(**number**) | estimating actual value(**random variable**) |

### Working-Hoteling Confidence Band

>[!note] Definition:
>Region $R\subseteq \mathbb{R}^2$ is a $(1-\alpha)100\%$ confidence band if
>$$
>\Pr(\beta_0+\beta_1X\subseteq R)= 1-\alpha\quad\text{for all }X
>$$

For SLR model, we can get Working-Hoteling Confidence Band as below:
$$
\text{For all values of }X\quad \hat{Y_h}\pm W\times SE(\hat{Y_h})
$$
where $W=\sqrt{2F(1-\alpha,2,n-2)}$ .


|           |                                         WH Band                                          |        CI        |
| --------- | :--------------------------------------------------------------------------------------: | :--------------: |
| Different | For all $X$, therefore need to **expand interval** to maintain $1-\alpha$ (WH Band > CI) | For only one $X$ |

### Joint CI for $g$ Mean Responses

#### Working-Hoteling Procedure

Same as [[Simple Linear Regression (SLR)#Working-Hoteling Confidence Band|Working-Hoteling Confidence Band]]
$$
\hat Y_h\pm W\cdot SE(\hat Y_h)
$$
where $W=\sqrt{2F(1-\alpha,2,n-2)}$ .

#### Bonferroni Procedure

$$
\hat Y_h\pm B\cdot SE(\hat Y_h)
$$
where $B=t(1-\frac{a}{2g},n-2)$ .
Actually, we can get a promotion from [[Simple Linear Regression (SLR)#Confidence Region of $( beta_0, beta_1)$|Confidence Region of ( beta_0, beta_1)]] 
$$
\Pr(\overset{p}{\underset{i=1}\cap}A_i^C)\geq 1-p\alpha
$$
where $A_i^C=\{\beta_i\notin CI_{1-\alpha}^i\}$ .

### Joint PI for $g$ New Observations

Predicting new observations $Y_h$ at $g$ different $X_h$'s
$$
\hat Y_h \pm B\cdot SE(prediction)
$$
where $B=t(1-\frac{a}{2g},n-2)$ .

### Departure from normality: $\varepsilon_i$ is not normal


|           | CI for $b_1$                                                      | HT for $b_1$                                                      | CI for $E\hat{Y_h}$                                                     | PI for $\hat{Y_h}$                                                                       |
| --------- | ----------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Large $n$ | By CLT, $b_1$ will be normal, CI is still **valid** for large $n$ | By CLT, $b_1$ will be normal, HI is still **valid** for large $n$ | By CLT, $\hat{Y_h}$ will be normal, CI is still **valid** for large $n$ | We require $Y_h$ normal, for large $n$ , deviation of PI is not rely on CLT, **invalid** |

# SLR Model in Matrix Notation

## Matrix Approach to Linear Regression

See lecture $9$ and Review Session Oct. $9$ .

## Matrix Expression

$$
\mathbf{Y}=\mathbf{X\beta+\varepsilon}=\begin{bmatrix}
1&X_1\\
\vdots&\vdots\\
1&X_n
\end{bmatrix}\begin{bmatrix}
\beta_0\\\beta_1
\end{bmatrix}
+\begin{bmatrix}
\varepsilon_1\\
\vdots\\
\varepsilon_n
\end{bmatrix}
$$
where $\mathbf{X}$ is **design matrix**, and $\varepsilon_i\overset{iid}{\sim}N(0,\sigma^2)$ or more succinctly as $\varepsilon\sim MVN(\mathbf{0},\sigma^2\mathbf{I}_n)$ .

## Least Squares

$$
\mathbf{b}=\begin{bmatrix}b_0\\b_1\end{bmatrix}=\underset{\beta\in\mathbb{R}^2}{\arg\min}\lVert\mathbf{Y}-\mathbf{X}\mathbf{\beta}\rVert^2=\underset{\beta\in\mathbb{R}^2}{\arg\min}\,Q(\beta)
$$
$$
\text{Normal Equation: } \mathbf{X}^T\mathbf{X}\mathbf{\beta}=\mathbf{X}^T\mathbf{Y}
$$

## Geometry

![Geometry](Geometry.jpg)
$\mathbf{H}$ is hat matrix, which is also the projection matrix onto column space of $\mathbf{X}$ . $\mathbf{H}$ only depends on $\mathbf{X}$, and it is a constant matrix.

## Properties of the hat matrix H

$$
\mathbf{H}:=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T
$$
$\mathbf{H}$ is idempotent matrix, which means
- $\mathbf{H}^2=\mathbf{H}$
- $\mathbf{H}$ is symmetric ($\mathbf{H}$ is orthogonal projection)
- $\mathbf{HX}=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X}=\mathbf{X}$ , so we have
$$
\mathbf{H1}=\mathbf{1},\,\mathbf{H}\mathbf{X}_1=\mathbf{X}_1,\cdots,\mathbf{H}\mathbf{X}_{p-1}=\mathbf{X}_{p-1}
$$
- $h_{ii}=\mathbf{H}_{ii}$, $0\leq h_{ii}\leq 1$ , $\sum_{i=1}^nh_{ii}=trace(\mathbf{H})=p$ ( trace(Projection matrix)=dim(Range) )

## Vector of residuals e

$$
\mathbf{e}=\mathbf{Y}-\mathbf{\hat Y}=(\mathbf{I}_n-\mathbf{H})\mathbf{Y}
$$
Easy to calculate
$$
E(\mathbf{e})=0\quad Var(\mathbf{e})=\sigma^2(\mathbf{I}_n-\mathbf{H})
$$
- $Var(e_i)=\sigma^2(1-\mathbf{H}_{i,i})\neq Var(\varepsilon_i)=\sigma^2$
- In general, $e_i$ are correlated since $\sigma^2(\mathbf{I}_n-\mathbf{H})$ is not diagonal

## Inference in Regression

$$
\begin{array}{}
Var(\mathbf{b})=\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}\\
s^2(\mathbf{b}_0)=\text{MSE}(\mathbf{X}^T\mathbf{X})^{-1}_{1,1}=\text{MSE}(\frac{1}{n}+\frac{\bar X^2}{S_{xx}})\\
s^2(\mathbf{b}_1)=\text{MSE}(\mathbf{X}^T\mathbf{X})^{-1}_{2,2}=\text{MSE}\frac{1}{S_{xx}}\\
\text{MSE}=\frac{1}{n-2}\|\mathbf{e}\|^2=\frac{1}{n-2}\mathbf{Y}^T(\mathbf{I}_n-\mathbf{H})\mathbf{Y}
\end{array}
$$

## Mean Response at Xh

$\hat{Y_h}=\mathbf{X}_h\mathbf{b}$, we have
$$
Var(\hat{Y}_h)=\sigma^2\mathbf{X}_h^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}_h=\sigma^2[\frac{1}{n}+\frac{(X_h-\bar X)^2}{S_{xx}}]
$$
