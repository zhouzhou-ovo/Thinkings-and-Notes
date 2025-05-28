For group comparisons, we have two models for two approaches
- Frequency: [[Frequency Statistics#ANOVA|ANOVA]]
- Bayesian: Hierarchical Model

## Comparing Two Groups

**Idea:** Information is shared between groups.

Let
$$
\begin{array}{c}
Y_{i,1}=\mu+\delta+\varepsilon_{i,1}\\
Y_{i,2}=\mu-\delta+\varepsilon_{i,2}\\
\varepsilon_{i,j}\overset{iid}{\sim}N(0,\sigma^2)
\end{array}
$$
Using this parameterization where $\theta_1=\mu+\delta$ and $\theta_2=\mu-\delta$, we see that $\delta$ represents half the population difference in means, as $(\theta_1-\theta_2)/2=\delta$, and $\mu$ represents the pooled average, as $(\theta_1+\theta_2)/2=\mu$.

## Convenient Analysis
### Conjugate Prior

Convenient conjugate prior distribution for the unknown parameters are
$$
\begin{array}{c}
p(\mu,\delta,\sigma^2)=p(\mu)\times p(\delta)\times p(\sigma^2)\\
\mu\sim N(\mu_0,\gamma_0^2)\\
\delta\sim N(\delta_0,\tau_0^2)\\
\sigma^2\sim\text{IG}(\frac{\nu_0}{2},\frac{\nu_0\sigma^2_0}{2})
\end{array}
$$

### Sampling Model

$$
\begin{array}{C}
Y_{i,1}|\mu,\delta,\sigma^2 \sim N(\mu+\delta,\sigma^2)\\
Y_{i,2}|\mu,\delta,\sigma^2\sim N(\mu-\delta,\sigma^2)
\end{array}
$$

### Full Conditional Posterior

$$
\begin{array}{rl}
\{ \mu \mid &y_1, y_2, \delta, \sigma^2 \} \sim \text{normal}(\mu_n, \gamma_n^2), \text{ where}\\

&\mu_n = \gamma_n^2 \times \left[ \mu_0 / \gamma_0^2 + \sum_{i=1}^{n_1} (y_{i,1} - \delta) / \sigma^2 + \sum_{i=1}^{n_2} (y_{i,2} + \delta) / \sigma^2 \right]\\


&\gamma_n^2 = \left[ 1 / \gamma_0^2 + (n_1 + n_2) / \sigma^2 \right]^{-1}\\[10pt]

\{ \delta \mid &y_1, y_2, \mu, \sigma^2 \} \sim \text{normal}(\delta_n, \tau_n^2), \text{ where}\\


&\delta_n = \tau_n^2 \times \left[ \delta_0 / \tau_0^2 + \sum_{i=1}^{n_1} (y_{i,1} - \mu) / \sigma^2 - \sum_{i=1}^{n_2} (y_{i,2} - \mu) / \sigma^2 \right]\\



&\tau_n^2 = \left[ 1 / \tau_0^2 + (n_1 + n_2) / \sigma^2 \right]^{-1}\\[10pt]


\{ \sigma^2 \mid &y_1, y_2, \mu, \delta \} \sim \text{inverse-gamma}(\nu_n / 2, \nu_n \sigma_n^2 / 2), \text{ where}\\


&\nu_n = \nu_0 + n_1 + n_2\\



&\nu_n \sigma_n^2 = \nu_0 \sigma_0^2 + \sum (y_{i,1} - [\mu + \delta])^2 + \sum (y_{i,2} - [\mu - \delta])^2

\end{array}
$$

## Hierarchical Normal Model

We are interested in $\theta_j$.

### Graphical Representation

![[graph hierarchical.png]]
We assume that $\sigma^2$ is constant variance.

### Semi-conjugate Prior

$$
\begin{array}{c}
\sigma^2\sim \text{IG}(\nu_0/2,\nu_0\sigma^2_0/2)\\
\tau^2\sim\text{IG}(\eta_0/2,\eta_0\tau_0^2/2)\\
\mu\sim N(\mu_0,\gamma_0^2)
\end{array}
$$
i.e. hyperpriors.
### Sampling Model

- Within-group model: $\phi_j=\{\theta_j,\sigma^2\}, p(y|\phi_j)=N(\theta_j,\sigma^2)$
- Between-group model: $\psi=\{\mu,\tau^2\},p(\theta_j|\psi)=N(\mu,\tau^2)$

### Posterior Inference

#### Joint Posterior Distributions

$$
\begin{array}{rl}
p(\theta_1,\cdots,\theta_,\mu,\tau^2,\sigma^2|\mathbf{y}_1,\cdots,\mathbf{y}_m)&\propto p(\mu,\tau^2,\sigma^2)\times p(\theta_1,\cdots,\theta_m|\mu,\tau^2,\sigma^2)\\
&\quad \times p(\mathbf{y}_1,\cdots,\mathbf{y}_m|\theta_1,\cdots,\theta_,\mu,\tau^2,\sigma^2)\\
&=p(\mu)p(\tau^2)p(\theta^2)\left\{\prod_{j=1}^m p(\theta_j|\mu,\tau^2) \right\}\left\{\prod_{j=1}^m\prod_{i=1}^{n_j} p(y_{i,j}|\theta_j,\sigma^2) \right\}\\
&\underset{\mu,\tau^2}{\propto} p(\mu)p(\tau^2)\prod_{j=1}^m p(\theta_j|\mu,\tau^2)
\end{array}
$$

#### Full Conditional Distributions of $\mu,\tau^2,\theta_j,\sigma^2$

$$
\begin{array}{rl}
\{\mu|\theta_1,\cdots,\theta_m,\tau^2\}&\sim N\left(\frac{m\bar\theta/\tau^2+\mu_0/\gamma_0^2}{m/\tau^2+1/\gamma_0^2},[m/\tau^2+1/\gamma_0^2]^{-1}\right)\\[10pt]

p(\mu|\theta_1,\cdots,\theta_m,\tau^2,\sigma^2,\mathbf{y}_1,\cdots,\mathbf{y}_m)&=p(\mu|\theta_1,\cdots,\theta_m,\tau^2)\\
&\propto p(\mu,\tau^2,\theta_1,\cdots,\theta_m)\\
&\propto p(\mu)\times \prod_{j=1}^m p(\theta_j|\mu,\tau^2)\\
&= Normal
\end{array}
$$

$$
\begin{array}{rl}
\{\tau^2|\theta_1,\cdots,\theta_m,\mu\}&\sim\text{IG}(\frac{\eta_0+m}{2},\frac{\eta_0\tau_0^2+\sum(\theta_j-\mu)^2}{2})\\[10pt]
p(\tau^2|\theta_1,\cdots,\theta_m,\mu,\mathbf{y}_1,\cdots,\mathbf{y}_m)&\propto p(\tau^2,\theta_1,\cdots,\theta_m,\mu,\mathbf{y}_1,\cdots,\mathbf{y}_m)\\
&\propto p(\tau^2)\prod_{j=1}^m p(\theta_j|\mu,\tau^2)\\
&=\text{IG}\times \text{Normal}\\
&=\text{IG}
\end{array}
$$

$$
\begin{array}{rl}
\{\theta_j|y_{1,j},\cdots,y_{n_j,j},\sigma^2\}&\sim N(\frac{n_j\bar{y}_j/\sigma^2+\mu/\tau^2}{n_j/\sigma^2+1/\tau^2},[n_j/\sigma^2+1/\tau^2]^{-1})\\[10pt]

p(\theta_j|\mathbf{y}_j,\sigma^2,\mu,\tau^2,\mathbf{y}_1,\cdots,\mathbf{y}_m)&\propto p(\theta_j|\mathbf{y}_j,\sigma^2,\mu,\tau^2)\\
&\propto p(\theta_j|\mu,\tau^2)\times \prod_{i=1}^{n_j}p(y_{i,j}|\theta_j,\sigma^2)\\
&= Normal
\end{array}
$$

$$
\begin{array}{rl}
\{\sigma^2|\boldsymbol{\theta},\mathbf{y}_1,\cdots,\mathbf{y}_m\}&\sim \text{IG}\left(\frac{1}{2}\left[\nu_0+\sum_{j=1}^mn_j\right],\frac{1}{2}\left[\nu_0\sigma^2_0+\sum_{j=1}^m\sum_{i=1}^{n_j}(y_{i,j}-\theta)^2\right]\right)\\[10pt]

p(\sigma^2|\boldsymbol{\theta},\mathbf{y}_1,\cdots,\mathbf{y}_m) &\propto p(\sigma^2,\boldsymbol{\theta},\mathbf{y}_1,\cdots,\mathbf{y}_m)\\
&\propto p(\sigma^2)\prod_{i=1}^m p(\mathbf{y}_i|\theta_i,\sigma^2)\\
&= \text{Inverse-Gamma}
\end{array}
$$

### Gibbs Sampling Procedure

For $\{\theta_1^{(s)},\cdots,\theta_m^{(s)},\mu^{(s)},\tau^{2(s)},\sigma^{2(s)}\}$, each scan we need to generate $m+3$ samples.

$$
\begin{array}{c}
\theta_j^{(s+1)}\sim p(\theta|\mu^{(s)},\tau^{2(s)},\sigma^{2(s)},\mathbf{y}_j)\quad j=1,\cdots,m\\
\sigma^{2(s)}\sim p(\sigma^2|\boldsymbol\theta^{(s+1)},\mathbf{y}_1,\cdots,\mathbf{y}_m)\\
\mu^{(s+1)}\sim p(\mu|\boldsymbol\theta^{(s+1)},\tau^{2(s)})\\
\tau^{2(s)}\sim p(\tau^2|\boldsymbol\theta^{(s+1)},\mu^{(s+1)})
\end{array}
$$

### Shrinkage

From full conditional distributions of parameters, we have known that
$$
E[\theta_j|\mathbf{y}_j]=\frac{n_j\bar{y}_j/\sigma^2+\mu/\tau^2}{n_j/\sigma^2+1/\tau^2}
$$
Notice that the expected value of $\theta_j$ is pulled a bit from $\bar y_j$ towards $\mu$ by an amount depending on $n_j$. This effect is called **shrinkage**.
![[shrinkage.png]]

>[!Note] Shrinkage对样本分析的作用
>![[shrinkage bayes vs freq.png]]
>
>通俗来讲，shrinkage描述的是随样本量增加，后验参数的值会向样本均值收缩。而这一特点对我们在进行小样本分析时很重要。
>
>频率学派在组间比较中会利用样本均值代表组特征从而进行分析。这种方法在大样本的情况下具有非常好的效果，但是在小样本情况下就会产生较大的误差。小样本情况下样本均值较小，不代表总体均值同样很小，可能因为采样的不同产生这种差异性。
>
>而为了防止这种情况带来的不准确性，贝叶斯学派注意到了“收缩”的特性，因此在小样本推断中我们会使用后验参数估计来代表组特征，而不是使用样本均值。

## Hierarchical Model with Non-constant Variance

![[hm non-constant var.png]]

---