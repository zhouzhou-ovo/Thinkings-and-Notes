
# Semi-conjugate Prior 

$$
p(\boldsymbol{\theta},\Sigma)=p(\boldsymbol{\theta})p(\Sigma)
$$
## Prior for Mean

We consider
$$
p(\theta)=MVN(\boldsymbol{\mu}_0,\Lambda_0)
$$
Therefore, we can have
$$
\begin{array}{rl}
p(\theta)\propto \exp\{-\frac{1}{2}\boldsymbol{\theta}^T\Lambda_0\boldsymbol{\theta}+\boldsymbol{\theta}^T\Lambda^{-1}_0\boldsymbol{\mu}\}\\
=\exp\{-\frac{1}{2}\boldsymbol{\theta}^T\mathbf{A}_0\boldsymbol{\theta}+\boldsymbol{\theta}^T\mathbf{b}_0\}
\end{array}
$$
where $\mathbf{A}_0=\Lambda_0^{-1},\mathbf{b}_0=\Lambda_0^{-1}\boldsymbol{\mu}_0$. We can also get
$$
\Lambda_0=\mathbf{A}_0^{-1}\quad \boldsymbol{\mu}_0=\Lambda_0\mathbf{b_0}=\mathbf{A}_0^{-1}\mathbf{b}_0
$$

## Prior for Covariance Matrix

### The Inverse-Wishart Distribution

1. sample $\mathbf{z}_1,\cdots,\mathbf{z}_{\nu_0}\overset{iid}{\sim}MVN(\mathbf{0},\boldsymbol{\Phi}_0)$ 
2. $\sum_{i=1}^{\nu_0}\mathbf{z}_i\mathbf{z}_i^T=$ sample from $\text{Wishart}(\nu_0,\boldsymbol{\Phi}_0)$
3. if we use sample covariance matrix $\mathbf{S}=\frac{1}{\nu_0}\sum_{i=1}^{\nu_0}\mathbf{z}_i\mathbf{z}_i^T$, then we have $\nu_0\mathbf{S}\sim \text{Wishart}(\nu_0,\boldsymbol{\Phi}_0)$

#### Fun Fact

$$
\sum_{i=1}^{\nu_0}\mathbf{z}_i\mathbf{z}_i^T=\mathbf{Z}^T\mathbf{Z}=(\mathbf{z}_1,\cdots,\mathbf{z}_n)(\mathbf{z}_1,\cdots,\mathbf{z}_n)^T
$$

Therefore, we have
$$
\begin{array}{c}
\Sigma^{-1}\sim \text{Wishart}(\nu_0,\mathbf{S}_0^{-1})\\
\Sigma \sim \text{Inverse-Wishart}(\nu_0,\mathbf{S}_0^{-1})
\end{array}
$$
with $E[\Sigma^{-1}]=\nu_0\mathbf{S}_0^{-1},E[\Sigma]=\frac{1}{\nu_0-p-1}\mathbf{S}_0$. $p$ is the dimension of $\mathbf{z}_i$. And the density of $\Sigma$ is
$$
\begin{array}{rl}

p(\Sigma) &= 

\left[

2^{\nu_0 p / 2} \pi^{p(p-1)/4} |S_0|^{-\nu_0 / 2}

\prod_{j=1}^p \Gamma\left(\frac{\nu_0 + 1 - j}{2}\right)

\right]^{-1}\\

&\times |\Sigma|^{-(\nu_0 + p + 1) / 2} \exp\left\{-\frac{1}{2} \mathrm{tr}(S_0 \Sigma^{-1})\right\}.
\end{array}
$$

## Sampling Model

$$
\mathbf{Y}_1,\cdots,\mathbf{Y}_n|\boldsymbol{\theta},\Sigma\sim MVN(\boldsymbol{\theta},\Sigma)
$$
Then we have
$$
p(\mathbf{Y}_1,\cdots,\mathbf{Y}_n|\boldsymbol{\theta},\Sigma)\propto \exp\{-\frac{1}{2}\boldsymbol{\theta}^T\mathbf{A}_1\boldsymbol{\theta}+\boldsymbol{\theta}^T\mathbf{b}_1\}
$$
where $\mathbf{A}_1=n\Sigma^{-1},\mathbf{b}_1=n\bar{\mathbf{y}}\Sigma^{-1}$. ($\bar{\mathbf{y}}=\frac{1}{n}(\sum_{i=1}^ny_{i,1},\cdots,\sum_{i=1}^ny_{i,p})^T$)

## Full Conditional Posterior

### Posterior for Mean
$$
\begin{array}{rl}
p(\boldsymbol{\theta}|\mathbf{y}_1,\cdots,\mathbf{y}_n,\Sigma)&\propto p(\mathbf{y}_1,\cdots,\mathbf{y}_n|\boldsymbol{\theta},\Sigma)\\
&\propto \exp\{-\frac{1}{2}\boldsymbol{\theta}^T(\mathbf{A}_0+\mathbf{A}_1)\boldsymbol{\theta}+\boldsymbol{\theta}^T(\mathbf{b}_0+\mathbf{b}_1)\}
\end{array}
$$
Therefore, we have
$$
\boldsymbol{\theta}|\mathbf{y}_1,\cdots,\mathbf{y}_n,\Sigma\sim MVN(\mathbf{A}_n^{-1}\mathbf{b}_n,\mathbf{A}_n^{-1})
$$
where
- $\mathbf{A}_n=\mathbf{A}_0+\mathbf{A}_1=\Lambda_0^{-1}+n\Sigma^{-1}$ means that posterior precision = prior precision + data precision
- $\mathbf{b}_n=\mathbf{b}_0+\mathbf{b}_1=\Lambda_0^{-1}\boldsymbol{\mu}_0+n\bar{\mathbf{y}}\Sigma^{-1}\Rightarrow$ posterior mean $\mathbf{A}_n^{-1}\mathbf{b}_n$ = weighted sum of $\boldsymbol{\mu}_0$ and $\bar{\mathbf{y}}$ = $(\Lambda_0^{-1}+n\Sigma^{-1})(\Lambda_0^{-1}\boldsymbol{\mu}_0+n\bar{\mathbf{y}}\Sigma^{-1})$  

### Posterior for Variance

$$
\begin{array}{rl}
p(\Sigma|\boldsymbol{\theta},\mathbf{y}_1,\cdots,\mathbf{y}_n)&\propto p(\Sigma)\times p(\mathbf{y}_1,\cdots,\mathbf{y_n}|\boldsymbol{\theta},\Sigma)\\
&\propto |\Sigma|^{-(\nu_0+n+p+1)/2}\exp\{-\text{tr}([\mathbf{S}_0+\mathbf{S}_\theta]\Sigma^{-1})/2\}\\
&(\text{use property of trace:}\text{ tr}(\mathbf{BAB}^T)=\text{tr}(\mathbf{B}^T\mathbf{BA}))
\end{array}
$$
where $\mathbf{S}_\theta=\sum_{i=1}^n(\mathbf{y}_i-\boldsymbol{\theta})(\mathbf{y}_i-\boldsymbol{\theta})^T$.

Therefore, we have
$$
\Sigma|\boldsymbol{\theta},\mathbf{y}_1,\cdots,\mathbf{y}_n\sim \text{IW}(\nu_n,\mathbf{S}_n^{-1})
$$
where
- $\nu_n=\nu_0+n$ means that posterior sample size = prior size + data sample size
- $\mathbf{S}_n=\mathbf{S}_0+\mathbf{S}_\theta=\mathbf{S}_0+\sum_{i=1}^n(\mathbf{y}_i-\boldsymbol{\theta})(\mathbf{y}_i-\boldsymbol{\theta})^T$ means that posterior residual sum of squares = prior residual s.s. + data residual s.s.

## Gibbs Sampler

Step 1
$$
\boldsymbol{\theta}^{(s+1)}\sim p(\boldsymbol{\theta}|\mathbf{y}_1,\cdots,\mathbf{y}_n,\Sigma^{(s)})
$$
- compute $\boldsymbol{\mu}_n,\Lambda_n$ from $\mathbf{y}_1,\cdots,\mathbf{y}_n,\Sigma^{(s)}$
- sample $\boldsymbol{\theta}^{(s+1)}\sim MVN(\boldsymbol{\mu}_n,\Lambda_n)$ 

Step 2
$$
\Sigma^{(s+1)}\sim p(\Sigma|\boldsymbol{\theta}^{(s+1)},\mathbf{y}_1,\cdots,\mathbf{y}_n)
$$
- compute $\mathbf{S}_n$ from $\mathbf{y}_1,\cdots,\mathbf{y}_n,\boldsymbol{\theta}^{(s+1)}$
- sample $\Sigma^{(s+1)}\sim \text{Inverse-Wishart}(\nu_0+n,\mathbf{S}_n^{-1})$

## Gibbs Sampling with Missing Data

Let $\mathbf{O}_i=(O_1,\cdots,O_p)^T$ where
- $O_{i,j}=1$ implies that $Y_{i,j}$ is observed and not missing
- $O_{i,j}=0$ implies $Y_{i,j}$ is missing

We 'll assume that missing data are missing at random, meaning that $\mathbf{O}_i$ and $\mathbf{Y}_i$ are statistically independent and that the distribution of $\mathbf{O}_i$ does not depend on $\boldsymbol{\theta},\Sigma$.

We can get the sampling probability for the data from subject $i$
$$
\begin{array}{rl}
p(\mathbf{o}_i,\{y_{i,j}:o_{i,j}=1\})&=p(\mathbf{o}_i)\times p(\{y_{i,j}:o_{i,j}=1\}|\boldsymbol{\theta},\Sigma)\\
&=p(\mathbf{o}_i)\times\int\left\{p(y_{i,1},\cdots,y_{i,p}|\boldsymbol{\theta},\Sigma)(\prod_{y_{i,j}:o_{i,j}=0}dy_{i,j})\right\}
\end{array}
$$

We let
- $\mathbf{Y}_{\text{obs}}=\{y_{i,j}:o_{i,j}=1\}$, the data that we do observe
- $\mathbf{Y}_{\text{miss}}=\{y_{i,j}:o_{i,j}=0\}$, the data that we do not observe

We want
$$
p(\boldsymbol{\theta},\Sigma,\mathbf{Y}_{\text{miss}}|\mathbf{Y}_{\text{obs}})
$$
- sample $\boldsymbol{\theta}^{(s+1)}$ from $p(\boldsymbol{\theta}|\Sigma^{(s)},\mathbf{Y}_{\text{miss}}^{(s)},\mathbf{Y}_{\text{obs}})$
- sample $\Sigma^{(s+1)}$ from $p(\Sigma|\boldsymbol{\theta}^{(s+1)},\mathbf{Y}_{\text{miss}}^{(s)},\mathbf{Y}_{\text{obs}})$
- sample $\mathbf{Y}_{\text{miss}}^{(s+1)}$ from $p(\mathbf{Y}_{\text{miss}}|\Sigma,^{(s+1)},\boldsymbol{\theta}^{(s+1)},\mathbf{Y}_{\text{obs}})$
