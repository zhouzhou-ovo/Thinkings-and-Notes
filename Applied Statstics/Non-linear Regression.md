
Consider
$$
Y_i=f(\mathbf{X}_i,\boldsymbol{\gamma})+\varepsilon_i
$$
where $\mathbf{X}_i: q\times 1$ , $\boldsymbol{\gamma}: p\times 1$, $f(\cdot,\cdot)$ (a response function) is some function of predictors $\mathbf{X}_i$ and parameters $\boldsymbol{\gamma}$ .

Response function:
- linear (MLR)
- non-linear, but intrinsically linear
$$
\text{Exponential Regression Model: }Y_i=\gamma_0\exp(\gamma_1X_i)+\varepsilon_i
$$
	where $f=\gamma_0\exp(\gamma_1X_i)\Rightarrow \log f=\log\gamma_0+\gamma_1X_i=\tilde{\gamma}_0+\gamma_1X_i$ 
- non-linear
$$
\text{Logistic Regression Model: }Y_i=\frac{\gamma_0}{1+\gamma_1\exp(\gamma_2X_i)}+\varepsilon_i
$$

### How to obtain $\boldsymbol{\gamma}$?

#### OLS
Objective function:
$$
Q(\boldsymbol{\gamma})=\sum_{i=1}^n[Y_i-f(\mathbf{X}_i,\boldsymbol{\gamma})]^2\Rightarrow \mathbf{g}=\arg\min_{\boldsymbol{\gamma}}Q(\boldsymbol{\gamma})
$$

#### MLE

Assume $\varepsilon_i\overset{iid}{\thicksim}N(0,\sigma^2)$, we have
$$
L(\boldsymbol{\gamma},\sigma^2)=\frac{1}{(2\pi\sigma^2)^{n/2}}\exp\{-\frac{1}{2\sigma^2}\sum_{i=1}^n[Y_i-f(\mathbf{X}_i,\boldsymbol{\gamma})]^2\}
$$

- MLE & OLS give same answer under  $\varepsilon_i\overset{iid}{\thicksim}N(0,\sigma^2)$
- must be solved numerically

### Logistic Regression

$Y_i=$ qualitative / categorical $=$ only $2$ values

possible solution:
$$
Y_i=\begin{cases}
1,p=\pi_i(X_i)\\
0,p=1-\pi_i(X_i)
\end{cases}
$$
each $Y_i$ has its own probability of success $\pi_i$ depends on $X_i$, $\pi_i$ continuous

$\Rightarrow$ fit $\pi_i\thicksim X_i$, then $Y_i\thicksim Bernoulli(\pi_i(X_i))$, we consider to use SLR: $\hat\pi_i =b_0+b_1X_i$

Goal: find a function $\hat\pi_i=f(b_0+b_1X_i)$ such that $f(-\infty,+\infty)\rightarrow [0,1]$.

Idea: start with continuousness response variable $Y^c\rightarrow$ dichotomization (二分)

#### Probit Regression

**Example:**
$$
\begin{array}{}
X=\text{alcohol},Y=\begin{cases}
1,\text{pre term baby}\\
0,\text{full term baby}
\end{cases}
\end{array}
$$
suppose $Y_i^c=\beta_0^c+\beta_1^cx_i+\varepsilon_i^c,\varepsilon_i^c\thicksim N(0,\sigma_c^2)$ 

Dichotomization
$$
Y_i=\begin{cases}
1, Y_i^c\leq 38\text{ weeks, pre term}\\
0, Y_i^c>38\text{ weeks, full term}
\end{cases}
$$
$$
\begin{array}{rl}
\Pr(Y_i=1)=\pi_i&= \Pr(Y_i^c\leq 38)=\Pr(\beta_0^c+\beta_1^cx_i+\varepsilon_i^c\leq 38)\\[8pt]

&= \Pr(\varepsilon_i^c\leq 38-\beta_0^c-\beta_1^cx_i)\\[8pt]

&= \Pr(\frac{\varepsilon_i^c}{\sigma_c}\leq\frac{38-\beta_0^c-\beta_1^cx_i}{\sigma_c})\\[8pt]

&= \Pr(z\leq \beta_0^*+\beta_1^*x_i)\\[8pt]

&= \Phi(\beta^*_0+\beta_1^*x_i)

\end{array}
$$
We just show: $\pi_i=\Pr(Y_i=1)=\Phi(\beta_0^*+\beta_1^*x_i)=E(Y_i)$ 

$\Phi$ is the function we want.
$$
E(Y_i)=\pi_i=\Phi(\beta_0^*+\beta_1^*x_i)\Leftrightarrow \pi_i^{-1}=\Phi^{-1}(\beta_0^*+\beta_1^*x_i)=\beta_0^*+\beta_1^*x_i
$$

Probit transform: $\Phi^{-1}:[0,1]\rightarrow [-\infty,\infty]$ 
![[probit regression plot.jpg]]

| MLR                                                                                                                                        | Probit                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathbf{Y}=\mathbf{X}\boldsymbol{\beta}+\boldsymbol{\varepsilon}$<br>$\boldsymbol{\varepsilon}\thicksim N(\mathbf{0},\sigma^2\mathbf{I})$ | $\Pr(Y_i=1)=\Phi(\mathbf{X}_i\boldsymbol{\beta})$                                                                                                                     |
| $Y_i\thicksim N(E(\mathbf{Y}_i,\sigma^2)$<br>$E(Y_i)=\mathbf{X}_i^T\boldsymbol{\beta}$                                                     | $Y_i\thicksim Bernoulli(E(Y_i))\leftarrow$ different random distribution<br>$E(Y_i)\thicksim \Phi(\mathbf{X}_i^T\boldsymbol{\beta})\leftarrow$ different distribution |

#### Simple Logistic Regression

Still dichotomization but error term is [[Probability Reference#Logistic Distribution|logistic R.V.]]
$$
Y_i^c=\beta_0^c+\beta_1^cx_i+\varepsilon_i^c
$$
where $\varepsilon_i^c\thicksim Logistic(mean=0,variance=\sigma_L^2)$

Same as [[Non-linear Regression#Probit Regression|Probit Regression]], we can get
$$
E(Y_i)=\Pr(Y_i=1)=F_L(\beta_0^*+\beta_1^*x_i)
$$
$$
E(Y_i)=\pi_i=F_L(\beta_0^*+\beta_1^*x_i)\Leftrightarrow \pi'_i=F_L^{-1}(\pi_i)=\beta_0^*+\beta_1^*x_i\,(\text{Logit Transform})
$$

Logistic Regression has better interpretation of $\beta_1^*$.

We define odds as
$$
odd_i=\frac{\pi_i}{1-\pi_i}=\frac{\Pr(Y_i=1)}{\Pr(Y_i=0)}\Leftrightarrow \log odd_i=\log\frac{\pi_i}{1-\pi_i}=F^{-1}_L(\pi_i)
$$

| SLR                                                              | Logistic                                                             |
| ---------------------------------------------------------------- | -------------------------------------------------------------------- |
| $E(Y_i)=\beta_0+\beta_1x_i$<br>$Y_i\thicksim N(E(Y_i),\sigma^2)$ | $E(Y_i)=F_L(\beta_0+\beta_1x_i)$<br>$Y_i\thicksim Bernoulli(E(Y_i))$ |

![[logistic plot.jpg]]

Log-log mean response function
$$
\pi'_i=\log(-\log(1-\pi_i))=F_G^{-1}(\pi_i)=\beta_0+\beta_1x_i
$$
error term $=$ [[Probability Reference#Gumbel Distribution|Gumbel Random Variable]]

Logistic regression is the most common one.

##### Estimate $\beta_0,\beta_1$ via MLE

From $\pi_i$ above, we can get
$$
f(Y_i)=\pi_i^{Y_i}(1-\pi_i)^{1-Y_i}
$$
Given $Y_1,\cdots,Y_n$ iid, we have
$$
\begin{array}{rl}
\text{Likelihood: }L(Y_1,\cdots,Y_n\vert \beta_0,\beta_1)&=\prod_{i=1}^n\pi_i^{Y_i}(1-\pi_i)^{1-Y_i}\\[8pt]

\text{log-likelihood: }l(Y_1,\cdots,Y_n\vert \beta_0,\beta_1)&=\sum_{i=1}^n[Y_i\log \pi_i\\
&\quad+(1-Y_i)\log(1-\pi_i)]\\[8pt]

&=\sum_{i=1}^n[Y_i\log\frac{\pi_i}{1-\pi_i}\\&\quad+\log(1-\pi_i)]\\[8pt]

&= \sum_{i=1}^n[Y_i(\beta_0+\beta_1x_i)\\&\quad-\log(1+\exp(\beta_0+\beta_1x_i))]
\end{array}
$$
maximize numerically $\Rightarrow$ $\hat\beta_0^{MLE}=b_0,\hat\beta_1^{MLE}=b_1$ 

fitted value: $\hat\pi_i=F_L(b_0+b_1x_i)=\text{estimated ean of }Y_i$ 

##### Interpretation of $b_1$

fitted log-odds at $X_i$ : $\hat\pi'_i(x_i)=b_0+b_1x_i$
fitted log-odds at $X_i+1$: $\hat\pi'_i(x_i+1)=b_0+b_1(x_i+1)$
$$
\begin{array}{rl}
\Rightarrow b_1 &= \hat\pi'_i(x_i+1)-\hat\pi'_i(x_i)\\
&= \text{change in fitted log-odds when }X\text{ increases by }1\\
&= \log\hat{\operatorname{odds}}(x_i+1)-\log\hat{\operatorname{odds}}(x_i)\\
&= \log \frac{\hat{\operatorname{odds}}(x_i+1)}{\hat{\operatorname{odds}}(x_i)}
\end{array}
$$
So we define
$$
\text{odd-ratio: }OR=\frac{\operatorname{odds}(x_i+1)}{\operatorname{odds}(x_i)}\text{ for any }X_i
$$
Therefore, we have
$$
b_1=\log \hat{OR}\Leftrightarrow \text{odds-ratio}=\exp(b_1)
$$

**As $X_i$ increases by $1$, odds get multipled by $\exp(b_1)$.**

#### Multiple Logistic Regression

$$
\begin{array}{rl}
E(Y_i)=\pi_i&=\frac{\exp(\mathbf{X}_i^T\boldsymbol{\beta})}{1+\exp(\mathbf{X}_i^T\boldsymbol{\beta})}=[1+\exp(-\mathbf{X}_i^T\boldsymbol{\beta})]^{-1}\\
&\Downarrow\\
\pi'_i=\log odds&=\beta_0+\beta_1x_{i1}+\cdots+\beta_{p-1}x_{i,p-1}
\end{array}
$$
![[multiple logistic.jpg]]
##### Inference

Wald test(z-test)
	For large samples,
	$$
	H_0:\beta_k=0\quad v.s. \quad H_a:\beta_k\neq 0
	$$
	Test statistics:
	$$
	|z^*|=\vert\frac{b_k}{SE(b_k)}|,\begin{cases}
	|z^*|>z(1-\alpha/2)\Rightarrow H_a\\
	|z^*|<z(1-\alpha/2)\Rightarrow H_0
	\end{cases}
	$$
#### CI for the Odds Ratio (OR)

$(1-\alpha) 100\%$ CI for $\beta_k$
$$
b_k\pm z(1-\alpha/2)\cdot SE(b_k)
$$
$(1-\alpha) 100\%$ CI for $OR=\exp(b_k)$
$$
\exp[b_k\pm z(1-\alpha/2)\cdot SE(b_k)]
$$
CI for mean response of $\mathbf{X}_h$
$$
\text{For }\pi_h'=\mathbf{X}_H^T\mathbf{b}\Rightarrow \pi_h'\pm z(1-\alpha/2)\sqrt{\mathbf{X}_H^Ts^2(\mathbf{b})\mathbf{X}_H}
$$
where $\sqrt{\mathbf{X}_H^Ts^2(\mathbf{b})\mathbf{X}_H}=SD(\mathbf{X}_H^T\mathbf{b})$ 
$$
\begin{array}{rl}
\text{For }\pi_h=F_L(\pi_h')&\Rightarrow F_L[\pi_h'\pm z(1-\alpha/2)\sqrt{\mathbf{X}_H^T\mathbf{b}\mathbf{X}_H}]\\
&=[F_L(\text{lower}),F_L(\text{upper})]\\
&=[(1+e^{-\text{lower}})^{-1},(1+e^{-\text{upper}})^{-1}]
\end{array}
$$

#### Polytomous/Multinomial Logistic Regression

$Y_i$ has $3$ or more outcomes.
$$
Y_{ij}=\begin{cases}
1,\text{ cat }j\\
0,\text{ otherwise}
\end{cases}
$$
where $\sum_{j=1}^nY_{ij}=1$ always for each $i$.
$$
\pi_{ij}=\Pr(Y_{ij}=1)=\text{prob that cat j is selected in ith response}
$$
For each $i$, $\sum_{j=1}^n\pi_{ij}=1$.

##### Estimated Mean Response

$$
\hat\pi_{ij}=\frac{\exp(\mathbf{X}_i^T\mathbf{b}_j)}{1+\exp(\mathbf{X}_i^T\mathbf{b}_j)}
$$
We need to estimate $\mathbf{b}_1,\mathbf{b}_2,\cdots,\mathbf{b}_{j-1}$ $j-1$ different vectors for categories $1,\cdots,j-1$ .

For binomial, $j=2$ ($2$ categories) $\Rightarrow$ just $1$ vector $\mathbf{b}$ to estimate.

### Poisson Regression

$Y_i=$ count of "rare" events. Response $\in\{0,1,2,\cdots\}$.

>[!question]
>Why rare?
>
>**Ans:**
>$$
>Pois(\mu)\approx Binomial(n,p)
>$$
>as $n\rightarrow \infty,p\rightarrow 0$, $np\rightarrow \mu$.

$Y_i=$ independent [[Probability Reference#Poisson Distribution|Poisson R.V.'s]] with parameter $\mu_i$
$$
\mu_i=\mu(\mathbf{X}_i,\boldsymbol{\beta})\Rightarrow Y_i\thicksim Pois(\mu_i)
$$
where $\mu_i$ changes for each observation $i$.

##### 3 Options for $\mu_i=(\mathbf{X}_i,\boldsymbol{\beta})$

- $\mathbf{X}_i^T\boldsymbol{\beta}$
- $\exp(\mathbf{X}_i^T\boldsymbol{\beta})$
- $\log(\mathbf{X}_i^T\boldsymbol{\beta})$

### General Linear Models

- $Y_1,\cdots,Y_n$ are response variables from an exponential family R.V. (bernoulli, normal, gamma, poisson, ...)
- A linear predictor $\mathbf{X}_i^T\boldsymbol{\beta}=\beta_0+\beta_1X_{i1}+\cdots+\beta_{p-1}\mathbf{X}_{i,p-1}$ 
- A link function $\mathbf{X}_i^T\boldsymbol{\beta}=g(\mu_i)$
	where $\mu_i$ is mean of $Y_i$ ; it links the mean of $Y_i$ to the linear predictor $\mathbf{X}_i^T\boldsymbol{\beta}$
	- $g(\cdot)$ is monotonic and differentable
	- $Var(Y_i)=\sigma_i^2$ no necessary constant, but $\sigma_i^2=h(\mu_i)=f(g^{-1}(\mathbf{X}_i^T\boldsymbol{\beta}))$ is some explicit function of the mean.

![[e.g. glm.jpg]]

---