# Hierarchical Regression Model (Linear Mixed Effects Model)

![[Graphical Hierarchical Regression Model.png]]

Consider $Y_{ij}=$ math score of student $i$ in High School $j$, $X_{ij}=$ socioeconomic status (SES) of student $i$ in High School $j$. We have
$$
Y_{ij}=\boldsymbol{\beta}_j^T\mathbf{x}_{i,j}+\varepsilon_{ij},\,\varepsilon_{ij}\overset{iid}{\sim} N(0,\sigma^2)
$$
If we rewrite $\boldsymbol{\beta}$, then we have
$$
\boldsymbol{\beta}_j=\boldsymbol{\theta}+\boldsymbol{\gamma}_j
$$
where $\boldsymbol{\theta}$ is referred to as a **fixed effect** (constant), $\boldsymbol{\gamma}_1,\cdot,\boldsymbol{\gamma}_m$ are called **random effects** and
$$
\boldsymbol{\gamma}_1,\cdots,\boldsymbol{\gamma}_m\overset{iid}{\sim} MVN (\mathbf{0},\Sigma)
$$
Therefore, our model becomes
$$
Y_{ij}=\boldsymbol{\theta}^T\mathbf{z}_{ij}+\boldsymbol{\gamma}_{j}\mathbf{x}_{ij}+\varepsilon_{ij}
$$
where 
- $\mathbf{z}_{ij}$ are predictors that don't change between groups
- $\mathbf{x}_{ij}$ are predictors that change between groups

## Priors / Hyperpriors

- Prior is the distribution of model parameters
- Hyperprior is the distribution of prior parameters

$$
\begin{array}{c}
\boldsymbol{\theta}\sim MVN(\boldsymbol{\mu}_0,\Lambda_0)\\
\Sigma\sim\text{inverse-Wishart}(\eta_0,\mathbf{S}_0^{-1})\\
\sigma^2\sim\text{inverse-gamma}(\frac{\nu_0}{2},\frac{\nu_0\sigma^2_0}{2})
\end{array}
$$

## Sampling Model

$$
\boldsymbol{\beta}_1,\cdots,\boldsymbol{\beta}_m\overset{iid}{\sim}MVN(\boldsymbol{\theta},\Sigma)
$$

## Full Conditional Distributions

$$
\begin{aligned}
\boldsymbol{\beta}_j|\mathbf{y}_j,\mathbf{X}_j,\boldsymbol{\theta},\Sigma,\sigma^2&\sim MVN(m_j,V_j)\\
V_j=\text{Var}[\beta_j | \mathbf{y}_j, \mathbf{X}_j, \sigma^2, \boldsymbol{\theta}, \boldsymbol{\Sigma}]

&= (\boldsymbol{\Sigma}^{-1} + \mathbf{X}_j^T \mathbf{X}_j / \sigma^2)^{-1} \\

m_j=\mathbb{E}[\beta_j | \mathbf{y}_j, \mathbf{X}_j, \sigma^2, \boldsymbol{\theta}, \boldsymbol{\Sigma}]

&= (\boldsymbol{\Sigma}^{-1} + \mathbf{X}_j^T \mathbf{X}_j / \sigma^2)^{-1} (\boldsymbol{\Sigma}^{-1} \boldsymbol{\theta} + \mathbf{X}_j^T \mathbf{y}_j / \sigma^2).

\end{aligned}
$$

$$
\begin{array}{c}
\{\boldsymbol{\theta} | \boldsymbol{\beta}_1, \dots, \boldsymbol{\beta}_m, \boldsymbol{\Sigma} \} 

\sim \text{MVN}(\boldsymbol{\mu}_m, {\Lambda}_m), \text{ where}\\

{\Lambda}_m = ({\Lambda}_0^{-1} + m {\Sigma}^{-1})^{-1}\\

\boldsymbol{\mu}_m = {\Lambda}_m ({\Lambda}_0^{-1} \boldsymbol{\mu}_0 + m {\Sigma}^{-1} \bar{\boldsymbol{\beta}})

\end{array}
$$

$$
\begin{array}{c}
\Sigma|\boldsymbol{\theta},\boldsymbol{\beta}_1,\cdots,\boldsymbol{\beta}_m\sim \text{inverse-Wishart}(\eta_0+m,[\mathbf{S}_0+\mathbf{S}_\theta]^{-1})
\end{array}
$$
where $\mathbf{S}_\theta=\sum_{j=1}^m(\boldsymbol{\beta}_j-\boldsymbol{\theta})(\boldsymbol{\beta}_j-\boldsymbol{\theta})^T$.

$$
\sigma^2\sim\text{inverse-gamma}(\frac{[\nu_0+\sum n_j]}{2},\frac{[\nu_0\sigma^2_0+SSR]}{2})
$$
where $SSR=\sum_{j=1}^m\sum_{i=1}^{n_j}(y_{ij}-\boldsymbol{\beta}^T_j\mathbf{x}_{ij})^2$.

# Generalized Linear Mixed Effects Models (Hierarchical GLM)

![[Hierarchical GLM.png]]

## Within Sampling Model

$$
p(\mathbf{y}_j|\boldsymbol{\beta}^T_j\mathbf{x}_{ij},\gamma)=\prod_{i=1}^{n_j}p(y_{i,j}|\boldsymbol{\beta}^T_j\mathbf{x}_{ij},\gamma)
$$
For example, we can have Poisson Regression, i.e.
$$
y_{i,j}|\boldsymbol{\beta}^T_j\mathbf{x}_{ij}\sim Pois(\lambda=\exp[\boldsymbol{\beta}^T_j\mathbf{x}_{ij}])
$$
and we have link function
$$
\log(E[y_{i,j}|\boldsymbol{\beta}^T_j\mathbf{x}_{ij}])=\boldsymbol{\beta}^T_j\mathbf{x}_{ij}
$$

## Metropolis-Hasting Approximation

**Gibbs Steps for $(\boldsymbol{\theta},\Sigma)$**

- Sample $\boldsymbol{\theta}^{(s+1)}$ from its full conditional distribution
- Sample $\Sigma^{(s+1)}$ from its full conditional distribution

**Metropolis Steps for $\boldsymbol{\beta}_j$**

![[Hierarchical GLM Metropolis Steps.png]]
