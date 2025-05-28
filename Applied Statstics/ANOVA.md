
### One-factor ANOVA

$Y$ is continuous response variable, $X$ is qualitative variable(predictor), i.e. **factor** (All predictors are categorical predictor)

level of a factor $=$ treatment $=$ a set of values of $X$ $\{X_1,\cdots,X_n\}$

$Y_{ij}=$ response of the $j$ th replicate(observation) to $i$ th treatment

| Treatment $1$                    | $\cdots$ | Treatment $r$                    |
| -------------------------------- | -------- | -------------------------------- |
| $Y_{11},Y_{12},\cdots,Y_{1,n_1}$ | $\cdots$ | $Y_{r1},Y_{r2},\cdots,Y_{r,n_r}$ |
where $n_i=$ number of replicate in treatment $i$, $n_T=\sum_{i=1}^rn_i=$ total number of observations.

#### Connection to Regression

Using dummy coding ($r$ levels $\Rightarrow$ $r-1$ variables)

>[!question] What are we interested in?
>Are all $\mu_i$ the same? i.e. does treatment have an effect on $Y$?

#### Cell Means Model(C.M.M.)

$Y_{ij}=$ response of the $j$ th replicate(observation) to $i$ th treatment
$$
\begin{array}{rl}
Y_{ij}&=\mu_i+\varepsilon_{ij},\quad\varepsilon_{ij}\overset{iid}{\thicksim}N(0,\sigma^2)\\[8pt]
\mu_i &=\text{cell mean}\\[8pt]
i&=1,\cdots,r\quad r=\text{number of treatment}\\[8pt]
j&=1,\cdots,n_i\quad n_i=\text{number of replicates in group/treatment }i
\end{array}
$$

See also in [[Bayesian Reference#Hierarchical Model|Hierarchical Model]].

##### Properties

| Treatment $1$                    | $\cdots$ | Treatment $r$                    |
| -------------------------------- | -------- | -------------------------------- |
| $Y_{11},Y_{12},\cdots,Y_{1,n_1}$ | $\cdots$ | $Y_{r1},Y_{r2},\cdots,Y_{r,n_r}$ |
| mean $\mu_1$                     | $\cdots$ | mean $\mu_r$                     |

$Y_{ij}=\mu_i+\varepsilon_{ij}\Rightarrow E(Y_{ij})=\mu_i,Var(Y_{ij})=\sigma^2\Rightarrow Y_{ij}\thicksim N(\mu_i,\sigma^2)$ 

##### As a Linear Model $\mathbf{Y}=\mathbf{X}\boldsymbol{\beta}+\boldsymbol{\varepsilon}$

$$
\begin{bmatrix}
Y_{11}\\\vdots\\Y_{1,n_1}\\\vdots\\Y_{r1}\\\vdots\\Y_{r,n_r}
\end{bmatrix}=\mathbf{Y}=\boldsymbol{X\beta}+\boldsymbol{\varepsilon}=\begin{bmatrix}
\mathbf{1}_{n_1}&&\\
&\ddots&\\
&&\mathbf{1}_{n_r}
\end{bmatrix}_{n_T\times r}\begin{bmatrix}
\mu_1\\\mu_2\\\vdots\\\mu_r
\end{bmatrix}+\begin{bmatrix}
\varepsilon_{11}\\\vdots\\\varepsilon_{1,n_1}\\\vdots\\\varepsilon_{r1}\\\vdots\\\varepsilon_{r,n_r}
\end{bmatrix}
$$

|           | CMM                             | Dummy Coding                                                                |
| --------- | ------------------------------- | --------------------------------------------------------------------------- |
| Intercept | ❌                               | ✔️                                                                          |
| Model     | $Y_{ij}=\mu_i+\varepsilon_{ij}$ | $Y_{ij}=\beta_0+\beta_1x_{i1}+\cdots+\beta_{r-1}x_{i,r-1}+\varepsilon_{ij}$ |
Connection: $\begin{cases}\mu_i=\beta_0+\beta_{i},i\neq r\\\mu_r=\beta_0\end{cases}$

##### Least-squares

$$
\begin{array}{rl}
Y_{i\cdot}&= \sum_{j=1}^{n_i}Y_{ij}=\text{sum of replicates for ith treatment}\\[8pt]

\bar Y_{i\cdot}&=\frac{1}{n_i}\sum_{j=1}^{n_i}Y_{ij}=\frac{1}{n_{i\cdot}}Y_{i\cdot}=\text{avg of }n_i\text{ obs in ith treatment}\\[8pt]

Y_{\cdot\cdot}&=\sum_{i=1}^r\sum_{j=1}^{n_i}Y_{ij}=\text{sum of all replicates across treatments}\\[8pt]

\bar Y_{\cdot\cdot}&=\frac{1}{n_T}Y_{\cdot\cdot}=\frac{1}{n_T}\sum_{i=1}^r\sum_{j=1}^{n_i}Y_{ij}=\sum_{i=1}^r\frac{n_i}{n_T}\bar Y_{i\cdot}=\text{weighted avg of treatment avg}
\end{array}
$$
We need to estimate $\mu_1,\cdots,\mu_r,\sigma^2$.

Objective function:
$$
\begin{array}{rl}
Q(\mu_1,\cdots,\mu_r)&=\sum_{i=1}^r\sum_{j=1}^{n_i}(Y_{ij}-\mu_i)^2\\[8pt]
&=\sum_j(Y_{1j}-\mu_1)^2+\cdots+\sum_j(Y_{rj}-\mu_r)^2
\end{array}
$$
Minimize w.r.t. $\mu_i$, we can get
$$
\hat\mu^{LS}_i=\bar Y_{i\cdot}\quad \hat Y_{ij}=\bar Y_{i\cdot}\quad e_{ij}=Y_{ij}-\hat Y_{ij}=Y_{ij}-\bar Y_{i\cdot}
$$
We can verify that
$$
\sum_{j=1}^{n_i}e_{ij}=0\text{ for each }i\Rightarrow \sum_{i=1}^r\sum_{j=1}^{n_i}e_{ij}=0
$$
##### MLE: Maximum Likelihood Estimation

Since $Y_{ij}=\mu_i+\varepsilon_{ij}\Rightarrow Y_{ij}\thicksim N(\mu_i,\sigma^2)$
$$
L(\mu_1,\cdots,\mu_r,\sigma^2)=\frac{1}{(2\pi\sigma^2)^{n_T/2}}\exp[-\frac{1}{2\sigma^2}\sum_i\sum_j(Y_{ij}-\mu_i)^2]
$$
Maximize $L$ w.r.t $\mu_i,\sigma^2$, we have
$$
\hat\mu_i^{MLE}=\hat\mu_i^{LS}=\bar Y_{i\cdot}\quad \hat{\sigma^2}^{MLE}=\frac{1}{n_T}SSE(\text{biased estimator})
$$

##### ANOVA Identity

$$
Y_{ij}-\bar Y_{\cdot\cdot}=Y_{ij}-\bar Y_{i\cdot}+\bar Y_{i\cdot}-\bar Y_{\cdot\cdot}
$$
We have
$$
\begin{array}{rl}
\sum_{i,j}(Y_{ij}-\bar Y_{ij})^2&=\sum_{i,j}(Y_{ij}-\bar Y_{i\cdot})^2+\sum_{i,j}(\bar Y_{i\cdot}-\bar Y_{\cdot\cdot})^2\\
&\quad+2\sum_{i,j}(Y_{ij}-\bar Y_{i\cdot})(\bar Y_{i\cdot}-\bar Y_{\cdot\cdot})
\end{array}
$$
We can prove that $\sum_{i,j}(Y_{ij}-\bar Y_{i\cdot})(\bar Y_{i\cdot}-\bar Y_{\cdot\cdot})=0$.

Then we get
$$
\sum_{i,j}(Y_{ij}-\bar Y_{ij})^2=\sum_{i,j}(Y_{ij}-\bar Y_{i\cdot})^2+\sum_{i,j}(\bar Y_{i\cdot}-\bar Y_{\cdot\cdot})^2
$$
- $SST_o=\sum_{i,j}(Y_{ij}-\bar Y_{ij})^2$: total variation of data around the mean $\bar Y_{\cdot\cdot}$
- $SSE=\sum_{i,j}(Y_{ij}-\bar Y_{i\cdot})^2$: deviation around estimated treatment mean (**within group variation**)
- $SSR=\sum_{i,j}(\bar Y_{i\cdot}-\bar Y_{\cdot\cdot})^2$: deviation estimated treatment mean around overall mean (**between group variation**)

![[aligned dot plot.jpg]]
##### ANOVA Table

| SS      | df      | MS                        | $E(MS)$                                                                                                  |
| ------- | ------- | ------------------------- | -------------------------------------------------------------------------------------------------------- |
| $SSE$   | $n_T-r$ | $MSE=\frac{SSE}{n_T-r}$   | $E(MSE)=\sigma^2$, MSE is unbiased estimation of $\sigma^2$                                              |
| $SST_r$ | $r-1$   | $MST_r=\frac{SST_r}{r-1}$ | $E(MST_r)=\sigma^2+\frac{\sum_in_i(\mu_i-\mu_0)^2}{r-1}$, where $\mu_0=\sum_{i=1}^r\frac{n_i}{n_T}\mu_i$ |
| $SST_o$ | $n_T-1$ |                           |                                                                                                          |

- If $\mu_1=\cdots=\mu_r\Rightarrow \mu_i=\mu_0\Rightarrow E(MST_r)=\sigma^2$ i.e. $MSE\approx MST_r$
- If at least $\mu_i\neq\mu_j\Rightarrow E(MST_r)>E(MSE)\Rightarrow MST_r\overset{>}{\sim} MSE$  

##### Hypothesis Test

$$
H_0:\mu_1=\mu_2=\cdots=\mu_r\quad v.s.\quad H_a:\text{at least one }\mu_i\text{ is dfferent}
$$
We can get test statistic
$$
F^*=\frac{MST_r}{MSE}=\frac{SST_r/(r-1)}{SSE/(n_T-r)}\thicksim F(r-1,n_T-r)
$$
Decision Rule:
- If $F^*>F(1-\alpha,r-1,n_T-r)\Rightarrow H_a$
- If $F^*<F(1-\alpha,r-1,n_T-r)\Rightarrow H_0$

Another sight, consider [[Simple Linear Regression (SLR)#General Linear Test|General Linear Test]], we have
$$
\begin{array}{rl}
\text{Reduced Model: }Y_{ij}=\mu_c+\varepsilon_{ij}\\
\text{Full Model: }Y_{ij}=\mu_1X_{ij1}+\cdots+\mu_rX_{ijr}+\varepsilon_{ij}
\end{array}
$$
We have test statistic
$$
\begin{array}{rl}
F^*&=\frac{SSE(R)-SSE(F)}{SSE(F)}\times\frac{df_F}{df_R-df_F}\\
&=\frac{SST_o-SSE}{SSE(F)}\times\frac{n_T-r}{n_T-1-(n_T-r)}\\
&=\frac{SST_r}{SSE}\times\frac{n_T-r}{r-1}\thicksim F(r-1,n_T-r)
\end{array}
$$

##### Inference for Factor Level Means

From [[ANOVA#Cell Means Model(C.M.M.)#Hypothesis Test|Hypothesis Test of CMM]], our conclusion is one-sided, we need to consider some deeper problems

- Which $\mu_i\neq\mu_j$?
- What's the CI for $\mu_i$?
- What's the CI for $\mu_i-\mu_j$?

###### Estimation for a Single Factor Level Mean $\mu_i$

$$
\hat\mu_i=\bar Y_{i\cdot}=\frac{1}{n_i}\sum_{j=1}^{n_i}Y_{ij}\thicksim N(\mu_i,\frac{\sigma^2}{n_i})
$$
where $Y_{ij}\overset{iid}{\thicksim}N(\mu_i,\sigma^2)$, $\sigma^2$ is unknown. We have
$$
\hat\sigma^2=S^2=MSE=\frac{SSE}{n_T-r}=\frac{\sum_{i,j}(Y_{ij}-\bar Y_{i\cdot})^2}{n_T-r}
$$
$$
\frac{\bar Y_{i\cdot}-\mu_i}{S(\bar Y_{i\cdot})}\thicksim t(n_T-r)$$
$$
(1-\alpha)100\%\,CI\,\text{for }\mu_i:[\bar Y_{i\cdot}\pm t(1-\alpha/2,n_T-r)\times S(\bar Y_{i\cdot})] 
$$
where $S(\bar Y_{i\cdot})=\sqrt{\frac{MSE}{n_i}}$ for $i=1,\cdots,r$.

###### Hypothesis Testing for a Single Factor Level Mean $\mu_i$

$$
H_0:\mu_i=c\quad v.s.\quad H_a:\mu_i\neq c
$$
Test statistic:
$$
t^*=\frac{\bar Y_{i\cdot}-c}{S(\bar Y_{i\cdot})}\thicksim t(n_T-r)\text{ under }H_0
$$
Decision Rules:
- If $|t^*|\leq t(1-\frac{\alpha}{2},n_T-r)\Rightarrow H_0$
- If $|t^*|>t(1-\frac{\alpha}{2},n_T-r)\Rightarrow H_a$

###### Estimation for the Difference of 2 Factor Level Means

Define
$$
D=\mu_i-\mu_j\Rightarrow \hat D=\hat\mu_i-\hat\mu_j\thicksim N(\mu_i-\mu_j,\frac{\sigma^2}{n_i}+\frac{\sigma^2}{n_j})
$$
$$
S^2(\hat D)=\hat\sigma^2(\frac{1}{n_i}+\frac{1}{n_j})=MSE(\frac{1}{n_i}+\frac{1}{n_j})
$$
Then we have
$$
\frac{\hat D-D}{S(\hat D)}\thicksim t(n_T-r)
$$
$(1-\alpha)100\%$ CI for $D$:
$$
[\hat D\pm t(1-\frac{\alpha}{2},n_T-r)\times S(\hat D)]
$$

###### Hypothesis Test for the Difference of 2 Factor Level Means

$$
H_0:D=0(\mu_i=\mu_j)\quad v.s. \quad H_a:D\neq 0(\mu_i\neq \mu_j)
$$
Test statistic:
$$
t^*=\frac{\hat D}{S(\hat D)}\thicksim t(n_T-r)\text{ under }H_0
$$
Decision Rules:
- if $|t^*|\leq t(1-\frac{\alpha}{2},n_T-r)\Rightarrow H_0$
- if $|t^*|> t(1-\frac{\alpha}{2},n_T-r)\Rightarrow H_a$

>[!Note] Coding for CMM
>- If we want to fit CMM in R, we need to use factor() for predictor and '-1' in lm()
>- If we want to get correct ANOVA table, we also need to use factor() for predictor

```R
g <- lm(y~factor(group)-1, data=df)
# -1 represents no intercept

g <- aov(y~factor(group),data=df)
summary(g)

aov(y~factor(group),data=df)
# These 2 ways can get correct ANOVA table.
```

###### Multiple Comparisons

Consider one-way ANOVA
$$
Y_{ij}=\mu_i+\varepsilon_{ij}
$$
E.g. $r=3$
$$
\begin{array}{rl}
T_1:H_0:\mu_1=\mu_2\quad v.s. H_a:\mu_1\neq \mu_2\\
T_2:H_0:\mu_1=\mu_3\quad v.s. H_a:\mu_1\neq \mu_3\\
T_3:H_0:\mu_3=\mu_2\quad v.s. H_a:\mu_3\neq \mu_2\\
\end{array}
$$
Like [[Simple Linear Regression (SLR)#Joint estimation of $( beta_0, beta_1)$|joint estimation of regression coefficients]], if we let $\alpha=\Pr(\text{reject }H_0|H_0\text{ is true})=0.05$, then we consider three test all hold, we have

$$
\begin{array}{rl}
\Pr(\text{accepting }H_0\text{ in }T_1,T_2,T_3|\text{all }H_0\text{'s are true})&=\prod_{i=1}^3\Pr(\text{accepting }H_0 \text{ in }T_i|H_0\text{ is true})\\[8pt]
&= (0.95)^3\\[8pt]
&\approx 0.857
\end{array}
$$

And we find

$$
\Pr(\text{reject at least one }H_0\text{ in }T_1,T_2,T_3|\text{all }H_0\text{'s are true})=1-0.857=0.143\approx 14\%
$$

We define $\text{Type I error}\overset{def}{=}\Pr(\text{reject }H_0\text{ given }H_0\text{ is true})$ and $\text{Type I error for }3\text{ tests}=\text{"Family-wise error"}$

If each test is at $\alpha=0.05\Rightarrow$ F.W.E $\approx 0.14\Rightarrow$ need to adjust $\alpha$ s.t. F.W.E.$\leq 0.05$ .

**Tukey Multiple Comparison Procedure**

- **Studentized Range Distribution**
	Consider $i.i.d. \,Y_1,\cdots,Y_r\thicksim N(\mu,\sigma^2)$, we have Range: $W=\max(Y_i)-\min(Y_i)$ and estimate $\sigma^2$ with $s^2$(degree of freedom $\nu$), we define studentized range R.V.
$$
q(r,\nu)=\frac{\text{range}}{\hat{SD}}=\frac{w}{s}
$$
	where $r=\text{sample size},\nu=\text{degree of freedom in }s^2 \text{ estimate}$.

- **Simultaneous CI's for ALL possible differences**
	Assume $r$ levels, $\mu_1,\cdots,\mu_r\leftarrow$ cell means, estimated all means $\bar Y_1,\cdots,\bar Y_r$. There are $\binom{r}{2}=\frac{r(r-1)}{2}$ differences we need to estimate.
	
	Consider simultaneous CI for $D=\mu_i-\mu_j$ for all $\binom{r}{2}$ pairs of $i,j$
	$$
	\hat D\pm T\times S(\hat D)
	$$
	where $\hat D=\hat\mu_i-\hat\mu_j,S(\hat D)=\sqrt{MSE(\frac{1}{n_i}+\frac{1}{n_j})},T=\frac{1}{\sqrt{2}}q(1-\alpha,r,n_T-r)$ 
	Intuition:
	$$
	\frac{\hat D-D}{S(\hat D)}=\frac{(\bar Y_{i\cdot}-\mu_i)-(\bar Y_{j\cdot}-\mu_j)}{S(\hat D)}\leq \frac{\max(\bar Y_{k\cdot}-\mu_k)-\min(\bar Y_{k\cdot}-\mu_k)}{S(\hat D)}\thicksim\frac{1}{\sqrt{2}}q(r,n_T-r)
	$$

- **Simultaneous Hypothesis Testing**
	$\frac{r(r-1)}{2}$ tests: $H_0:\mu_i=\mu_j\quad vs \quad H_a:\mu_i\neq \mu_j$ for all $i\neq j$ 
	
	Test statistic: $q^*=\frac{\sqrt{2}\hat D}{S(\hat D)}\thicksim q(r,n_T-r)$ 
	
	Decision Rule:
	- $|q^*|\leq q(1-\alpha,r,n_T-r)\Rightarrow H_0$
	- $|q^*|> q(1-\alpha,r,n_T-r)\Rightarrow H_a$

- **Simultaneous CI (Bonferroni)**
	$$
	\hat D\pm B\times S(\hat D)\quad B=t(1-\frac{\alpha}{2g},n_T-r)
	$$
	where $g=\frac{r(r-1)}{2}$ is the number of comparisions.
	
	Usually Bonferroni too conservative (CI's too wide), Tukey intervals are narrower.
	
	**But if $g<\frac{r(r-1)}{2}$, then Bonferroni could be narrower than Tukey. 

#### Factor Effect Model(F.E.M.)

Consider $\mu_i=\mu_0+(\mu_i-\mu_0)=\mu_0+\tau_i$, where $\mu_0$ is constant component fro all treatments, $\tau_i$ is ith treatment(factor) effect. We have
$$
Y_{ij}=\mu_{\cdot}+\tau_i+\varepsilon_{ij},\quad i=1,\cdots,r-1,\quad j=1,\cdots,n_i
$$
##### As a Linear Model $\mathbf{Y}=\mathbf{X}\boldsymbol{\beta}+\boldsymbol{\varepsilon}$

$$
\begin{bmatrix}
Y_{11}\\
\vdots\\
Y_{1,n_1}\\
\vdots\\
Y_{r-1,1}\\
\vdots\\
Y_{r-1,n_{r-1}}\\
Y_{r1}\\
\vdots\\
Y_{r,n_r}
\end{bmatrix}=\mathbf{Y}=\boldsymbol{X\beta}+\boldsymbol{\varepsilon}=\begin{bmatrix}
1&1&0&\cdots&0&0\\
\vdots&\vdots&\vdots&&\vdots&\vdots\\
1&1&0&\cdots&0&0\\
\vdots&\vdots&\vdots&&\vdots&\vdots\\
1&0&0&\cdots&0&1\\
\vdots&\vdots&\vdots&&\vdots&\vdots\\
1&0&0&\cdots&0&1\\
1&-1&-1&\cdots&-1&-1\\
\vdots&\vdots&\vdots&&\vdots&\vdots\\
1&-1&-1&\cdots&-1&-1\\
\end{bmatrix}\begin{bmatrix}
\mu_\cdot\\
\tau_1\\
\vdots\\
\tau_{r-1}
\end{bmatrix}+\begin{bmatrix}
\varepsilon_{11}\\\vdots\\\varepsilon_{1,n_1}\\\vdots\\\varepsilon_{r1}\\\vdots\\\varepsilon_{r,n_r}
\end{bmatrix}
$$
$$
Y_{ij}=\mu_\cdot+\tau_1X_{ij1}+\cdots+\tau_{r-1}X_{ij,r-1}+\varepsilon_{ij}
$$
where $X_{ijq}=\begin{cases}1,\text{if treat }q\\-1,\text{if treat }r\\0,\text{ow.}\end{cases}$ , $q=1,\cdots,r-1$, $\mu$ is intercept, $\tau_i$ is slopes

##### Least-squares

$$
\hat\mu_\cdot=\frac{1}{r}\sum_{i=1}^r\bar Y_{i\cdot}\quad \hat \tau_i=\bar Y_{i\cdot}-\hat \mu_\cdot
$$

##### Choice for $\mu_{\cdot}$ 

- $\mu_{\cdot}=\frac{1}{r}\sum_{i=1}^r\mu_i\Rightarrow \sum_{i=1}^r\tau_i=0$ 
- $\mu_{\cdot}=\sum_{i=1}^rw_i\mu_i$ (factor effect with weighted mean)

##### Hypothesis Test

$$H_0:\tau_1=\cdots=\tau_{r-1}=0\quad v.s.\quad H_a:\text{at least one }\tau_i\neq 0$$
We can test via overall model fit F-test
$$
F^*=\frac{MST_r}{MSE}=\frac{SST_r/df_{T_r}}{SSE/df_{SSE}}\thicksim F(r-1,n_T-r)
$$
We can also use [[Simple Linear Regression (SLR)#General Linear Test|General Linear Test]],
$$
\begin{array}{rl}
\text{Reduced Model: }Y_{ij}&=\mu_\cdot+\varepsilon_{ij}\\
\text{Full Model: }Y_{ij}&=\mu_\cdot+\tau_1X_{ij1}+\cdots+\tau_{r-1}X_{ij,r-1}+\varepsilon_{ij}
\end{array}
$$
and we can get the same statistic as above.

|            | CMM                                                              | FEM                                                                |
| ---------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| Hypothesis | $H_0:\mu_1=\cdots=\mu_r$<br>$H_a:$ at least one $\mu_i\neq\mu_j$ | $H_0:\tau_1=\cdots=\tau_r=0$<br>$H_a:$ at least one $\tau_i\neq 0$ |
| Variables  | $r$ variables, no $=0$                                           | $r-1$                                                              |

### Two-Factor ANOVA

$Y=$ continuous response variable, we investigate how $Y$ depends on two categorical/ qualitative variables.
![[two factor ANOVA.jpg]]

| $\text{Factor B}\backslash\text{Factor A}$ |          $i=1$           |          $i=2$           |          $i=3$           |
| :----------------------------------------: | :----------------------: | :----------------------: | :----------------------: |
|                   $j=1$                    | $Y_{111},Y_{112},\cdots$ | $Y_{211},Y_{212},\cdots$ | $Y_{311},Y_{312},\cdots$ |
|                   $j=2$                    | $Y_{121},Y_{122},\cdots$ | $Y_{221},Y_{222},\cdots$ | $Y_{321},Y_{322},\cdots$ |

#### Effects Model

$$
Y_{ijk}=\mu_{\cdot\cdot}+\alpha_i+\beta_j+(\alpha\beta)_{ij}+\varepsilon_{ijk}
$$
where
- $\mu_{\cdot\cdot}:$ constant part of all cells
- $\alpha_i:$ main effect of $A$ at level $i$
- $\beta_j:$ main effect of $B$ at level $j$
- $(\alpha\beta)_{ij}:$ not product, one coefficient, interaction of $A,B$ at level $i,j$
- $\alpha_i+\beta_j+(\alpha\beta)_{ij}:$ effect of $i,j$

![[scenario 1-2 of two way anova.jpg]]
