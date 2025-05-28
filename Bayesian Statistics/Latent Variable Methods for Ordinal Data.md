![[Graphic latent.pic.jpg]]

We consider $3$ variables that
- $\text{DEG}_i=$ highest degree obtained by individual $i$
- $\text{CHILD}_i=$ number of children they have
- $\text{PDEG}_i=$ binary, whether or not either parent of $i$ obtained a college degree

We want to measure how educational attainment is related to the number of children and the education of parents.

If we use a linear regression model
$$
\text{DEG}_i=\beta_1+\beta_2\times \text{CHILD}_i+\beta_3\times \text{PDEG}_i+\beta_4\times \text{CHILD}_i\times\text{PDEG}_i+\varepsilon_i
$$
where we assume that $\varepsilon_i\overset{i.i.d.}{\sim}N(0,\sigma^2)$, we'll meet some problems
- **Non-normal**: Since the variable $\text{DEG}$ takes on only a small set of discrete values, the normality assumption of the residuals will certainly be violated.
- **Scale**: The regression model imposes a numerical scale to the data that is not really present: A bachelor’s degree is not “twice as much” as a high school degree, and an associate’s degree is not “two less” than a graduate degree.

i.e. the discrete variables $\text{DEG},\text{CHILD}$ are **ordinal variables**(there is a logical ordering of the sample space).

We need another model to deal with these problems.

# Probit Regression

**Idea:** We relate a variable $Y$ to a vector of predictors $\mathbf{x}$ via a regression in terms of a **latent variable** $Z$.

We have the model
$$
\begin{array}{c}
\varepsilon_1,\cdots,\varepsilon_n\overset{i.i.d.}{\sim} N(0,1)\\
Z_i=\boldsymbol{\beta}^T\mathbf{x}_i+\varepsilon_i\\
Y_i=g(Z_i)
\end{array}
$$
where
- $Z_i$: latent random variable (unordered)
- $Y_i$: observations
- $g(\cdot)$: link function, **non-decreasing**

If the sample space for $Y$ takes on $K$ values, say $\{1,...,K\}$, then the function $g$ can be described with only $K − 1$ ordered parameters $g_1 < g_2 < \cdots <g_{K−1}$ as follows:
$$
y = g(z) =

\begin{cases} 

1 & \text{if } -\infty = g_0 < z < g_1 \\ 

2 & \text{if } g_1 < z < g_2 \\ 

\vdots & \\ 

K & \text{if } g_{K-1} < z < g_K = \infty.

\end{cases}
$$
We need to estimate parameters $\{\boldsymbol{\beta},g_1,\cdots,g_{K-1},\mathbf{Z}\}$.

## Prior

$$
\begin{array}{c}
\boldsymbol{\beta}\sim MVN(\mathbf{0},n(\mathbf{X}^T\mathbf{X})^{-1})\quad(\text{Zellner's g prior})\\
\end{array}
$$

## Sampling Model

$$
Z_i\sim N(\boldsymbol{\beta}^T\mathbf{x}_i,1)
$$

## Full Conditional Distributions

$$
\begin{array}{c}
p(\boldsymbol{\beta}|\mathbf{y},\mathbf{z},g)=p(\boldsymbol{\beta}|\mathbf{z})\propto p(\boldsymbol{\beta})p(\mathbf{z}|\boldsymbol{\beta})\\
\boldsymbol{\beta}|\mathbf{z}\sim MVN(\mathbf{m},\mathbf{V})\\
\mathbf{V}=\frac{n}{n+1}(\mathbf{X}^T\mathbf{X})^{-1})\\
\mathbf{m}=\frac{n}{n+1}(\mathbf{X}^T\mathbf{X})^{-1})\mathbf{X}^T\mathbf{z}
\end{array}
$$

$$
p(z_i|\boldsymbol{\beta},\mathbf{y},g)\propto \text{dnorm}(z_i,\boldsymbol{\beta}^T\mathbf{x}_i,1)\times \delta_{(a,b)}(z_i)
$$
where
$$
\delta_{(a,b)}(z_i)=\begin{cases}
1,\text{ if }z_i\in(a,b)\\
0,\text{ otherwise}
\end{cases}
$$
and $a=g_{y_i-1},b=g_{y_i}$. This is the density of a **constrained normal distribution**.

For full conditional distribution of $\mathbf{g}$, we first get constraints:
$$
\begin{array}{c}
g_k>z_i=g^{-1}(y_i=k)\Rightarrow g_k> a_k= \max\{z_i:y_i=k\}\\
g_k<z_i=g^{-1}(y_i=k+1)\Rightarrow g_k< b_k= \min\{z_i:y_i=k+1\}\\ 
Supp\{\mathbf{g}\}=\{\mathbf{g}:a_k<g_k<b_k\}
\end{array}
$$
Therefore, if $p(\mathbf{g})$ is normal, then we have
$$
p(g_k|\boldsymbol{\beta},\mathbf{y},\mathbf{g}_{-k})\propto \text{dnorm}(g_k,\mu_k,\sigma_k)\times\delta_{(a_k,b_k)}(g_k)
$$

We use Gibbs to approximate $p(\boldsymbol{\beta},\mathbf{g},\mathbf{z}|\mathbf{y},\mathbf{X})$.

We also find some issues with ordinal probit models:
- we don't know the prior of $g_k$
- there will be problems when the number of categorical is large (e.g. $K\geq20$)

# Transformation Models and the Rank Likelihood

**Idea:** if the $Z_i$'s were observed directly, then we would not have to estimate the transformation $g(z)$.

We know that $g$ is non-decreasing, we do know something about the order of the $Z_i$'s. Having observed $\mathbf{Y}=\mathbf{y}$, we know that the $Z_i$'s must lie in the set
$$
R(\mathbf{y})=\{z\in\mathbb{R}^n:z_{i_1}<z_{i_2}\text{ if }y_{i_1}<y_{i_2}\}
$$
Since the distribution of the $Z_i$’s does not depend on $g$, the probability that $Z \in R(y)$ for a given $y$ also does not depend on the unknown function $g$
$$
\begin{aligned}
p(\boldsymbol{\beta}|\mathbf{Z}\in R(\mathbf{y}))&\propto p(\boldsymbol{\beta})\times \Pr(\mathbf{Z}\in R(\mathbf{y})|\boldsymbol{\beta})\\
&=p(\boldsymbol{\beta})\times\int_{R(\mathbf{y})}\prod_{i=1}^n\text{dnorm}(z_i,\boldsymbol{\beta}^T\mathbf{x}_i,1)dz_i
\end{aligned}
$$
As a function of $\boldsymbol{\beta}$, the probability $\Pr(\mathbf{Z}\in R(\mathbf{y})|\boldsymbol{\beta})$ is known as the rank likelihood. It is called a rank likelihood because for continuous data it contains the same information about y as knowing the ranks of $\{y_1, \cdots , y_n\}$, i.e. which one has the highest value, which one has the second highest value, etc.
