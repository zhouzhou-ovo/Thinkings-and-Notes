## Normal Linear Regression Model

$$
\begin{array}{c}
\varepsilon_1,\cdots,\varepsilon_n\overset{iid}{\sim}N(0,\sigma^2)\\
Y_i=\boldsymbol{\beta}^T\mathbf{x}_i+\varepsilon_i\Rightarrow Y_i\sim N(\boldsymbol{\beta}^T\mathbf{x}_i,\sigma^2)
\end{array}
$$

## Bayesian Estimation

### Semi-conjugate Prior

$$
\begin{array}{c}
\boldsymbol{\beta}\sim MVN(\boldsymbol{\beta}_0,\Sigma_0)\\
\gamma=1/\sigma^2\sim \text{Gamma}(\nu_0/2,\nu_0\sigma^2_0/2)
\end{array}
$$

### Joint Sample Model

$$
\mathbf{y}|\mathbf{X},\boldsymbol{\beta},\sigma^2\sim MVN(\mathbf{X}\boldsymbol{\beta},\sigma^2\mathbf{I})
$$

### Full Conditional Posterior

$$
\begin{array}{rl}
p(\boldsymbol{\beta}|\mathbf{y},\mathbf{X},\sigma^2)&\propto p(\mathbf{y}|\boldsymbol{\beta},\mathbf{X},\sigma^2)p(\boldsymbol{\beta})\\
&\propto \exp[-\frac{1}{2}\boldsymbol{\beta}^T(\Sigma^{-1}_0+\frac{1}{\sigma^2}\mathbf{X}^T\mathbf{X})\boldsymbol{\beta}\\
&\quad+\boldsymbol{\beta}^T(\Sigma^{-1}_0\boldsymbol{\beta}_0+\frac{1}{\sigma^2}\mathbf{X}\mathbf{y})]
\end{array}
$$
Therefore, we have $\boldsymbol{\beta}|\mathbf{y},\mathbf{X},\sigma^2\sim MVN(\mathbf{m},\mathbf{V})$ where
- $\mathbf{V}=(\Sigma^{-1}_0+\frac{1}{\sigma^2}\mathbf{X}^T\mathbf{X})^{-1}$ means that posterior precision = prior precision + data precision
- $\mathbf{m}=(\Sigma^{-1}_0+\frac{1}{\sigma^2}\mathbf{X}^T\mathbf{X})^{-1}(\Sigma^{-1}_0\boldsymbol{\beta}_0+\frac{1}{\sigma^2}\mathbf{X}\mathbf{y})$, the second term means that prior precision $\times$ prior mean $+$ OLS estimate precision $\times$ OLS estimate
	- if $\sigma^2\rightarrow \infty, \mathbf{m}\rightarrow \boldsymbol{\beta}_0$
	- if $\Sigma_0\rightarrow \infty$ i.e. all eigenvalues $\rightarrow \infty,\mathbf{m}\rightarrow \hat{\beta}^{OLS}$

$$
\begin{array}{rl}
p(\gamma|\boldsymbol{\beta},\mathbf{y},\mathbf{X})&\propto p(\mathbf{y}|\boldsymbol{\beta},\mathbf{X},\gamma)\times p(\gamma)\\
&\propto \gamma^{\frac{n+\nu_0}{2}-1}\exp[-\frac{\gamma}{2}(SSR(\boldsymbol\beta)+\nu_0\sigma_0^2)]
\end{array}
$$
Therefore, we have $\sigma^2|\mathbf{y},\mathbf{X},\boldsymbol{\beta}\sim\text{IG}(\frac{n+\nu_0}{2},\frac{SSR(\boldsymbol\beta)+\nu_0\sigma_0^2}{2})$ where $SSR(\boldsymbol{\beta})$ is the sum of squares of residuals.

### Connection to the Frequentist Approach

If we set flat prior on $(\boldsymbol{\beta},\log\sigma^2)$, i.e. $p(\boldsymbol{\beta})\propto 1,p(\sigma^2)\propto \frac{1}{\sigma^2}$ (Jeffrey's Prior, improper), we can show
$$
\begin{array}{rl}
\boldsymbol{\beta}|\sigma^2,\mathbf{X},\mathbf{y}&\sim MVN(\hat{\beta}^{OLS},Var(\hat{\beta}^{OLS}))\\
&=MVN((\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y},\sigma^2(\mathbf{X}^T\mathbf{X})^{-1})\\
\sigma^2|\mathbf{X},\mathbf{y}&\sim \text{IG}(\frac{n-p}{2},\frac{n-p}{2}s^2)
\end{array}
$$
where $s^2=\frac{1}{n-p}SSR(\hat{\beta}^{OLS})$.

### Gibbs Sampler

1. updating $\boldsymbol{\beta}$:
	- Compute $\mathbf{V}=\text{Var}[\beta|\mathbf{y},\mathbf{X}, \sigma^{2(s)}]$ and $\mathbf{m} = \mathbb{E}[\beta \mid \mathbf{y}, \mathbf{X}, \sigma^{2(s)}]$
	- Sample $\beta^{(s+1)} \sim \text{multivariate normal}(\mathbf{m}, \mathbf{V})$
2. updating $\sigma^2$:
	- Compute $\text{SSR}(\beta^{(s+1)})$
	- Sample $\sigma^{2(s+1)} \sim \text{inverse-gamma}\left(\frac{\nu_0 + n}{2}, \frac{\nu_0 \sigma_0^2 + \text{SSR}(\beta^{(s+1)})}{2}\right)$

# Default and Weakly Informative Prior Distributions

## Unit Information Prior

**Idea:** $\text{Prec}(\hat\beta^{\text{OLS}})=\frac{\mathbf{X}^T\mathbf{X}}{\sigma^2}=$ precision of n observations $\Rightarrow$ on average, $1$ observation has observation $\frac{1}{n}(\frac{\mathbf{X}^T\mathbf{X}}{\sigma^2})=\Sigma_0^{-1}$

Let 
$$
\beta_0=\hat\beta^{\text{OLS}},\Sigma_0=n\sigma^2(\mathbf{X}^T\mathbf{X})^{-1},\nu_0=1,\sigma_0^2=s^2
$$
This is Empirical Bayes.

## Zellner's Prior

**Idea:** If predictors change in scale, the inference should not.
Let
$$
\boldsymbol{\beta}_0=\mathbf{0},\Sigma_0=g\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}
$$
Then we have
$$
\boldsymbol{\beta}|\mathbf{X},\mathbf{y},\sigma^2\sim MVN(\mathbf{m},\mathbf{V})\begin{cases}
\mathbf{m}=\frac{g}{g+1}\hat\beta^{\text{OLS}}\\
\mathbf{V}=\frac{g}{g+1}\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}
\end{cases}
$$
- $g=0$, strong prior $\Rightarrow \mathbf{m}=\boldsymbol{\beta}_0=\mathbf{0}$
- $g=n,\Sigma_0=n\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}\Rightarrow$ U.I.P.
- $g\rightarrow \infty$, weak prior $\Rightarrow \mathbf{m}\rightarrow \hat\beta^{\text{OLS}}$

# Model Selection: Bayesian Model Comparison

WWaFD: [[Frequency Statistics#Model Selection|Model Selection]]

We write the regression coefficient for variable $j$ as $\beta_j=z_j\times b_j$, where $z_j\in\{0,1\}$. Then we have regression equation
$$
y_i=z_1b_1x_{i,1}+\cdots+z_pb_px_{i,p}+\varepsilon_i
$$
We need to estimate: $\mathbf{z},\mathbf{b},\sigma^2|\mathbf{X},\mathbf{y}$.

In generally, if we want to compute posterior probability, we have
$$
p(\mathbf{z}|\mathbf{X},\mathbf{y})=\frac{p(\mathbf{z})p(\mathbf{y}|\mathbf{z},\mathbf{X})}{\sum_{\tilde{\mathbf{z}}}p(\tilde{\mathbf{z}})p(\mathbf{y}|\tilde{\mathbf{z}},\mathbf{X})}
$$
We can find if we have $p$ predictors, then the denominator has $2^p$ terms, which means when there are too many predictors, the computational cost will be very high.

Therefore, we can compare the evidence for any two models with the posterior [[Some Mathematical Ideas in Bayesian Statistics#7. odds 的引入|odds]]:
$$
\text{odds}(\mathbf{z}_a,\mathbf{z}_b|\mathbf{X},\mathbf{y})=\frac{p(\mathbf{z}_a|\mathbf{X},\mathbf{y})}{p(\mathbf{z}_b|\mathbf{X},\mathbf{y})}=\frac{p(\mathbf{z}_a)}{p(\mathbf{z}_b)}\times \frac{p(\mathbf{y}|\mathbf{X},\mathbf{z}_a)}{p(\mathbf{y}|\mathbf{X},\mathbf{z}_b)}
$$
which means posterior odds = prior odds $\times$ "Bayes factor".
- Bayes Factor measures how much the data favor model $\mathbf{z}_a$ over model $\mathbf{z}_b$

Hence, we only need to calculate $p(\mathbf{y}|\mathbf{X},\mathbf{z})$

$$
\begin{array}{rl}
p(\mathbf{y}|\mathbf{X},\mathbf{z})&=\iint p(\mathbf{y},\boldsymbol{\beta},\sigma^2|\mathbf{X},\mathbf{z})d\boldsymbol{\beta}d\sigma^2\\
&=\iint p(\mathbf{y}|\boldsymbol{\beta},\sigma^2,\mathbf{X},\mathbf{z})\times p(\boldsymbol{\beta},\sigma^2|\mathbf{X},\mathbf{z})d\boldsymbol{\beta}d\sigma^2\\
&=\iint p(\mathbf{y}|\boldsymbol{\beta},\sigma^2,\mathbf{X},\mathbf{z})p(\boldsymbol{\beta}|\mathbf{X},\mathbf{z})p(\sigma^2)d\boldsymbol{\beta}d\sigma^2\\
&=\iint \{\text{sampling model}\}\times \{\text{g-prior}\}
\times \{IG\}\\
&= K (1 + g)^{-p_z/2} \frac{(\nu_0\sigma^2_0)^{\nu_0/2}}{\left(\nu_0 \sigma_0^2 + \text{SSR}_g^z\right)^{\frac{\nu_0 + n}{2}}}
\end{array}
$$
where $K=\pi^{-n/2}\frac{\Gamma([\nu_0+n]/2)}{\Gamma(\nu_0/2)},p_z=\sum_{j=1}^pz_j$ and $SSR_g^z$ is the sum of squared residual for mode $\mathbf{z}$.

We can find
- as $p_z\uparrow,p(\mathbf{y}|\mathbf{X},\mathbf{z})\downarrow$
- as $p_z\uparrow,SSR_g^z\downarrow,p(\mathbf{y}|\mathbf{X},\mathbf{z})\uparrow$
which means we have balance between the number of predictor and the fit.

### Gibbs Sampling (Collapsed Gibbs Sampling)

Our goal is to estimate $p(z_1,\cdots,z_p|\mathbf{y},\mathbf{X})$, but if p is large, then it will be impractical for us to compute the marginal probability of each model.

In these situations our data analysis goals become more modest: For example, we may be content with a decent estimate of $\beta$ from which we can make predictions, or a list of relatively high-probability models.(也就是说我们会适当降低一些分析的标准，得到一个相对较好的模型即可)

Our Gibbs sampling scheme is
$$
\begin{matrix}
\uparrow\\
\mathbf{z}^{(s)}&\longrightarrow& \sigma^{2(s)}&\longrightarrow& \boldsymbol\beta^{(s)}\\
\uparrow\\
\vdots\\
\uparrow\\
\mathbf{z}^{(2)}\\
\uparrow\\
\mathbf{z}^{(1)}
\end{matrix}
$$
We can call this procedure as "Gibbs Elevator" 

Generating values of $\{\mathbf{z}^{(s+1)},\sigma^{2(s+1)},\boldsymbol{\beta}^{(s+1)}\}$ from $\mathbf{z}^{(s)}$ is achieved with the following steps:
1. Set $\mathbf{z}=\mathbf{z}^{(s)}$
2. For $j\in\{1,\cdots,p\}$ in random order, replace $z_j$ with a sample from $p(z_j|\mathbf{z}_{-j},\mathbf{y},\mathbf{X})$. We can calculate the probability due to conditional odds $o_j$

$$
\begin{array}{c}
o_j=\frac{\Pr(z_j=1|\mathbf{z}_{-j},\mathbf{y},\mathbf{X})}{\Pr(z_j=0|\mathbf{z}_{-j},\mathbf{y},\mathbf{X})}=\frac{\Pr(z_j=1)}{\Pr(z_j=0)}\times\frac{p(\mathbf{y}|\mathbf{z}_{-j},z_j=1,\mathbf{X})}{p(\mathbf{y}|\mathbf{z}_{-j},z_j=0,\mathbf{X})}\quad(\frac{\Pr(z_j=1)}{\Pr(z_j=0)}=1)\\[8pt]


\Pr(z_j=1|\mathbf{z}_{-j},\mathbf{y},\mathbf{X})=\frac{o_j}{1+o_j}
\end{array}
$$

1. Set $\mathbf{z}^{(s+1)}=\mathbf{z}$
2. Sample $\sigma^{2(s+1)}\sim p(\sigma^2|\mathbf{z}^{(s+1)},\mathbf{y},\mathbf{X})$
3. Sample $\boldsymbol{\beta}^{(s+1)}\sim p(\boldsymbol{\beta}|\sigma^{2(s+1)},\mathbf{z}^{(s+1)},\mathbf{y},\mathbf{X})$

### Prediction

1. $\mathbf{z}^*=\underset{\mathbf{z}}{\arg\max}\,p(\mathbf{z}|\mathbf{y},\mathbf{X})$ 
2. Bayesian Model Averaging (贝叶斯平均)
$$\hat{\boldsymbol{\beta}}_{BMA}=\frac{1}{S}\sum_{i=1}^S\boldsymbol{\beta}^{(i)}$$
	It is average over different models. BMA performs better in prediction.
