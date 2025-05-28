 

![[pgm normal model.jpg]]

# Properties of Normal Distribution

For normal distribution $p(y|\theta,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\{-\frac{1}{2}\left(\frac{y-\theta}{\sigma}\right)^2\},y\in(-\infty,\infty)$ , we have
- mean = median = mode = $\theta$
- symmetric around $\theta$
- $\Pr(Y\in[\theta\pm 1.96\sigma])\approx 0.95$ (This property will help us to identity prior parameters)

# Inference for the Mean (Conditional on the Variance)

## Sampling Model

$$
Y_1,\cdots,Y_n|\theta,\sigma^2\overset{iid}{\sim} N(\theta,\sigma^2)
$$
Based on joint sampling density, we can get two-dimensional sufficient statistic $\{\sum y_i,\sum y_i^2\}\Rightarrow\{\bar y,s^2\}$.

## Conditional Prior (Conjugate)

$$
\theta\sim N(\mu_0,\tau_0^2)
$$
or if the prior mean were based on $\kappa_0$ prior observations from the same (or similar) population as $Y_1,\cdots,Y_n$, then we set $\tau_0^2=\frac{\sigma^2}{\kappa_0}$, the variance of the mean of the prior observations:
$$
\theta\sim N(\mu_0,\frac{\sigma^2}{\kappa_0})
$$
## Conditional Posterior

[[Some Mathematical Ideas in Bayesian Statistics#6. Pro-Move 对比较复杂的参数进行换元简化计算|Pro-Move]]
$$
\begin{array}{rl}
p(\theta|y_1,\cdots,y_n,\sigma^2)&\propto p(y_1,\cdots,y_n|\theta,\sigma^2)p(\theta|\sigma^2)\\
&=N(\mu_n,\tau_n^2)
\end{array}
$$
where
$$
\begin{array}{rl}
\mu_n&=\frac{\tilde{\tau}_0^2}{\tilde{\tau}_0^2+n\tilde{\sigma}^2}\mu_0+\frac{n\tilde{\sigma}^2}{\tilde{\tau}_0^2+n\tilde{\sigma}^2}\bar y\\
&=\frac{\kappa_0}{\kappa_0+n}\mu_0
+\frac{n}{\kappa_0+n}\bar y\\
&=\frac{\kappa_0}{\kappa_n}\mu_0+\frac{n}{\kappa_n}\bar y
\end{array}
$$
$$
\frac{1}{\tau_n^2}=\tilde{\tau}_n^2=\tilde{\tau}_0^2+n\tilde{\sigma}^2=\frac{1}{\tau_0^2}+\frac{n}{\sigma^2}
$$
parameters with $\sim$ is **precision.** We can think about precision as the quantity of information. So we have posterior information = prior information + posterior information. 

>[!Note] 集中度与Information
>我们可以通过后验均值的表达式中发现，如果先验和样本模型中哪一方的精度（precision）更高，那么其在后验均值的加权平均中占据的权重就会更大。这也说明了，数据更加集中（也就是方差更小，精度更大）相比分散的数据会保留更多的信息，也就是说我们可以直观地得到数据的特征。

## Prediction

We have two methods to get the predictive distribution.

### Method 1: Decomposition of Predictive Variables

We have
$$
\tilde{Y}|\theta,\sigma^2\sim N(\theta,\sigma^2)\Leftrightarrow
 \tilde{Y}=\theta+\tilde{\varepsilon},\,\tilde\varepsilon|\theta,\sigma^2\sim N(0,\sigma^2) $$
Then we can calculate
$$
\begin{array}{rl}
E[\tilde{Y}|y_1,\cdots,y_n,\sigma^2]&=E[\theta+\tilde{\varepsilon}|y_1,\cdots,y_n,\sigma^2]\\
&= E[\theta|y_1,\cdots,y_n,\sigma^2]+E[\tilde{\varepsilon}|y_1,\cdots,y_n,\sigma^2]\\
&=\mu_n+0=\mu_n
\end{array}
$$
$$
\begin{array}{rl}
Var[\tilde{Y}|y_1,\cdots,y_n,\sigma^2]&=Var[\theta+\tilde{\varepsilon}|y_1,\cdots,y_n,\sigma^2]\\
&= Var[\theta|y_1,\cdots,y_n,\sigma^2]+Var[\tilde{\varepsilon}|y_1,\cdots,y_n,\sigma^2]\\
&=\tau_n^2+\sigma^2
\end{array}
$$
Therefore, we have 
$$
\tilde{Y}|\sigma,y_1,\cdots,y_n\sim N(\mu_n,\tau_n^2+\sigma^2)
$$

>[!Note] Intuition about the Form of the Variance of $\tilde{Y}$
>通常而言，对于新变量 $\tilde{Y}$ 的不确定性应该是一个关于 $\tau_n^2$ 和 $\sigma^2$ 的函数。随着$n\rightarrow \infty$，$\tau_n^2\rightarrow 0$，也就是说我们会对 $\theta$ 的位置越来越精确，但是这并不能减少样本方差 $\sigma^2$，因此我们对于$\tilde{Y}$的不确定性永远不会低于$\sigma^2$。

### Method 2: Law of Total Variance
 
 [[Some Mathematical Ideas in Bayesian Statistics#5. 条件期望的塔式性质/迭代法则，全方差定理(Law of Total Variance)与ANOVA|条件期望的塔式性质与Law of Total Variance]]

First, we need to identify what distribution $\tilde{Y}$ is.
$$
\begin{array}{rl}
p(\tilde{y}|y_1,\cdots,y_n,\sigma^2)&=\int p(\tilde{y},\theta|y_1,\cdots,y_n,\sigma^2)d\theta\\
&=\int p(\tilde y|y_1,\cdots,y_n,\theta,\sigma^2)p(\theta|y_1,\cdots,y_n,\sigma^2)d\theta\\
&=\int p(\tilde{y}|\theta,\sigma^2)p(\theta|y_1,\cdots,y_n,\sigma^2)d\theta\\
&= N(\text{mean},\text{variance})
\end{array}
$$
For mean, we have
$$
\begin{array}{rl}
E[\tilde{Y}|y_1,\cdots,y_n,\sigma^2]&=E[E[\tilde{Y}|y_1,\cdots,y_n,\sigma^2,\theta]|y_1,\cdots,y_n,\sigma^2]\\
&=E[\theta|y_1,\cdots,y_n,\sigma^2]\\
&=\mu_n
\end{array}
$$
For variance, we have
$$
\begin{array}{rl}
Var[\tilde{Y}|y_1,\cdots,y_n,\sigma^2]&=E[Var[\tilde{Y}|y_1,\cdots,y_n,\theta,\sigma^2]|y_1,\cdots,y_n,\sigma^2]\\
&\quad+ Var[E[\tilde{Y}|y_1,\cdots,y_n,\theta,\sigma^2]|y_1,\cdots,y_n,\sigma^2]\\
&=E[\sigma^2|y_1,\cdots,y_n,\sigma^2]\\
&\quad+Var[\theta|y_1,\cdots,y_n,\sigma^2]\\
&=\sigma^2+\tau_n^2
\end{array}
$$
where $\sigma^2$ describes uncertainty in $\tilde{Y}$ around $\theta$, $\tau_n^2$ describes uncertainty in $\theta$.

# Joint Inference for the Mean and Variance

We have known that $p(\theta,\sigma^2)=p(\theta|\sigma^2)p(\sigma^2)$, we need to find a family of prior distributions of $\sigma^2$ that has support on $(0,\infty)$. One choice is the gamma family, but **this family is not conjugate for the normal variance**. 

However, the gamma family does turn out to be a conjugate class of densities for precision $1/\sigma^2$. Then we have
$$
\begin{array}{c}
\text{precision}=1/\sigma^2\sim \text{gamma}(a,b)\\
\text{variance}=\sigma^2\sim \text{inverse-gamma}(a,b)
\end{array}
$$

## Conjugate Prior

$$
\begin{array}{c}
\theta|\sigma^2\sim N(\mu_0,\tau_0^2=\frac{\sigma^2}{\kappa_0})\\ \frac{1}{\sigma^2}\sim \text{inverse-gamma}(\frac{\nu_0}{2},\frac{\nu_0\sigma_0^2}{2})
\end{array}
$$
We can interpret the prior parameters $(\sigma_0^2,\nu_0)$ as the sample variance and sample size of prior observations.

## Sampling Model

$$
Y_1,\cdots,Y_n|\theta,\sigma^2\overset{iid}{
\sim}N(\theta,\sigma^2)
$$

## Posterior

From Conditional Posterior above, we can get
$$
\begin{array}{rl}
p(\theta,\sigma^2|y_1,\cdots,y_n)&=p(\theta|y_1,\cdots,y_n,\sigma^2)p(\sigma^2|y_1,\cdots,y_n)\\
&=N(\mu_n,\tau_n^2)\times p(\sigma^2|y_1,\cdots,y_n)
\end{array}
$$
For second term, we have
$$
\begin{array}{rl}
p(\sigma^2|y_1,\cdots,y_n)&\propto p(y_1,\cdots,y_n|\sigma^2)p(\sigma^2)\\
&=p(\sigma^2)\int p(y_1,\cdots,y_n,\theta|\sigma^2)d\theta\\
&=p(\sigma^2)\int p(y_1,\cdots,y_n|\theta,\sigma)p(\theta|\sigma^2)d\theta\\
&=\text{inverse-gamma}(\frac{\nu_n}{2},\frac{\nu_n\sigma^2_n}{2})
\end{array}
$$
where 
- $\nu_n=\nu_0+n\Rightarrow$ posterior sample size = prior sample size + data sample size
- $\nu_n\sigma_n^2=\nu_0\sigma_0^2+(n-1)s^2+\frac{\kappa_0n}{\kappa_0+n}(\bar y-\mu_0)^2\Rightarrow$ posterior sum of squares = prior s.s. + data sum of s.s. + extra term of estimate $\sigma^2$
- As $\kappa_0,\nu_0\rightarrow 0,\mu_n\rightarrow \bar y,\sigma^2_n\rightarrow \frac{n-1}{n}s^2=\frac{1}{n}\sum_i(y_i-\bar y)^2$.

Therefore, we have
$$
p(\theta,\sigma^2|y_1,\cdots,y_n)=N(\mu_n,\tau_n^2)\times \text{IG}(\frac{\nu_n}{2},\frac{\nu_n\sigma^2_n}{2})
$$
## Monte Carlo Sampling

$$
\begin{matrix}
\sigma^{2(1)}\sim \text{inverse-gamma}(\frac{\nu_n}{2},\frac{\nu_n\sigma^2_n}{2}),&&\theta^{(1)}\sim N(\mu_n,\frac{\sigma^{2(1)}}{\kappa_n})\\
\vdots&&\vdots\\
\sigma^{2(S)}\sim \text{inverse-gamma}(\frac{\nu_n}{2},\frac{\nu_n\sigma^2_n}{2}),&&\theta^{(S)}\sim N(\mu_n,\frac{\sigma^{2(S)}}{\kappa_n}).
\end{matrix}
$$

## Improper Prior

Let $\kappa_0,\nu_0\rightarrow 0$, we have $p(\theta,\sigma^2)\rightarrow N(\mu_0,\infty)\times \text{IG}(0,0)$ which is improper prior. But we can get proper posterior base on this prior
$$
\begin{array}{c}
\theta|\sigma^2,y_1,\cdots,y_n\sim N(\bar y,\frac{\sigma^2}{n})\\
\frac{1}{\sigma^2}|y_1,\cdots,y_n\sim\text{gamma}(\frac{n}{2},\frac{n}{2}\frac{1}{n}\sum_i(y_i-\bar y)^2)
\end{array}
$$

From the posterior, we can get
$$
\frac{\theta-\bar y}{s/\sqrt{n}}|y_1,\cdots,y_n\sim t_{n-1}\qquad\frac{\bar y-\theta}{s/\sqrt{n}}|\theta\sim t_{n-1}
$$
- The right term means before sampling data, uncertainty about the scale deviation of $\bar y$ from $\theta$ is represented with $t_{n-1}$
- The left term means after sampling data, uncertainty is still represented with $t_{n-1}$.

# Prior Specification Based on Expectations

The normal model is a two-dimensional exponential family model, where
- $\mathbf{t}(y)=(y,y^2)$
- $\boldsymbol{\phi}=(\frac{\theta}{\sigma^2},-\frac{1}{2\sigma^2})$
- $c(\boldsymbol{\phi})=|\phi_2|^{1/2}\exp\{\phi_1^2/(2\phi_2)\}$
- $\mathbf{t}_0=(E[Y],E[Y^2])$ ([[One-parameter Models#Conjugate Prior Distribution and Posterior Distribution|How to calculate]])

If we are given prior information on the mean as $E[\theta]=\mu_0$ and the variance $E[\sigma^2]=\sigma_0^2$ , we incorporate that into our prior that
$$
\begin{array}{rl}
p(\theta,\sigma^2)&\propto [(\sigma^2)^{-1/2}\exp\left\{\frac{-n_0(\theta-t_1)^2}{2\sigma^2}\right\}]\times\\
&\quad [(\sigma^2)^{-(n_0+5)/2}\exp\left\{\frac{-n_0(t_2-t_1^2)}{2\sigma^2}\right\}]\\
&=p(\theta|\sigma^2)p(\sigma^2)\\
&= N(\mu_0,\sigma^2/n)\times\text{IG}(\frac{n_0+3}{2},\frac{(n_0+1)\sigma^2_0}{2})
\end{array}
$$

If prior is weak, i.e. we set $n_0=1$, we can get our posterior
$$
\begin{array}{c}
\theta|\sigma^2,y_1,\cdots,y_n\sim N(\frac{\mu_0+n\bar y}{1+n},\frac{\sigma^2}{n+1})\\
\sigma^2|y_1,\cdots,y_n\sim \text{IG}(2+\frac{n}{2},\sigma_0^2+\frac{(n-1)s^2}{2}+\frac{n}{2(n+1)}(\bar y-\mu_0)^2)
\end{array}
$$

# The Normal Model for Non-normal Data

Based on **Central Limit Theorem**, we consider the sampling distribution of the **sample mean $\bar y$** instead of the sampling distribution of a singe data.

# Normal Model with Semi-conjugate Prior

In some situations, our certainty about $\theta$ depends on $\sigma^2$, i.e. $p(\theta|\sigma^2)$. But in others we may want to specify our certainty about $\theta$ as being independent of $\sigma^2$, i.e. $p(\theta,\sigma^2)=p(\theta)p(\sigma^2)$. One such joint distribution is the following semi-conjugate prior distribution.

## Semi-conjugate Prior

$$
\begin{array}{c}
p(\theta,\sigma^2)=p(\theta)p(\sigma^2)\\
\theta\sim N(\mu_0,\tau_0^2)\\
1/\sigma^2\sim \text{Gamma}(\nu_0/2,\nu_0\sigma_0^2/2)
\end{array}
$$

## Full Conditional Posterior

In Conditional Posterior, we have $\theta|\sigma^2,\mathbf{y}\sim N(\mu_n,\tau_n^2)$, where $\mu_n=\frac{\tilde{\tau}_0^2}{\tilde{\tau}_0^2+n\tilde{\sigma}^2}\mu_0+\frac{n\tilde{\sigma}^2}{\tilde{\tau}_0^2+n\tilde{\sigma}^2}\bar y$, $\tilde{\tau}_n^2=\tilde{\tau}_0^2+n\tilde{\sigma}^2$.

For full conditional posterior of $\sigma^2$, we have
$$
\begin{array}{rl}
p(\tilde{\sigma}^2|\theta,y_1,\cdots,y_n)&\propto p(\tilde{\sigma}^2,\theta,y_1,\cdots,y_n)\\
&=p(y_1,\cdots,y_n|\theta,\tilde{\sigma}^2)p(\theta|\tilde{\sigma}^2)p(\tilde{\sigma}^2)\\
&\propto p(y_1,\cdots,y_n|\theta,\tilde{\sigma}^2)p(\tilde{\sigma}^2)\\
&\propto \text{Gamma}(\nu_n/2,\nu_n\sigma^2(\theta)/2)
\end{array}
$$
where
$$
\nu_n=\nu_0+n,\quad \sigma^2(\theta)=\frac{1}{\nu_n}[\nu_0\sigma_0^2+ns_n^2(\theta)]=\frac{1}{\nu_n}[\nu_0\sigma_0^2+\sum_i(y_i-\theta)^2].
$$

>[!tip] 关于 $s_n^2(\theta)$ 的另一种形式
>在代码实现的过程中，我们可以将 $s_n^2(\theta)$ 转换为另一种形式
>$$
>s_n^2(\theta)=\sum_i(y_i-\bar y+\bar y-\theta)^2=(n-1)s^2+n(\bar y-\theta)^2
>$$
>这种形式可以令 Gibbs Sampling 的代码以更快的速度运行，因为 $s^2,\bar y$ 都是已知量，不需要重复计算。

## Discrete Approximation

In the semi-conjugate case we can just approximate the posterior distribution:
$$
\begin{array}{rl}
p(\theta,\tilde{\sigma}^2,y_1,\cdots,y_n)&=p(\theta,\tilde{\sigma}^2)p(y_1,\cdots,y_n|\theta,\tilde{\sigma}^2)\\
&= N(\mu_0,\tau_0^2)\times \text{Gamma}(\frac{\nu_0}{2},\frac{\nu_0\sigma_0^2}{2})\\
&\quad \times \prod_i N(\theta,1/\tilde{\sigma}^2)
\end{array}
$$

Letting $\{\theta_1,\cdots,\theta_G\},\{\tilde{\sigma}_1^2,\cdots,\tilde{\sigma}_H^2\}$ be sequences of **evenly** spaced parameter values, the discrete approximation to the posterior distribution assigns a posterior probability to each $\{\theta_k,\tilde{\sigma}_l^2\}$ on the grid, given by
$$
\begin{array}{rl}
p_D(\theta_k,\tilde{\sigma}_l^2|y_1,\cdots,y_n)&=\frac{p(\theta_k,\tilde{\sigma}_l^2|y_1,\cdots,y_n)}{\sum_{g=1}^G\sum_{h=1}^Hp(\theta_g,\tilde{\sigma}_h^2|y_1,\cdots,y_n)}\\
&=\frac{p(\theta_k,\tilde{\sigma}_l^2,y_1,\cdots,y_n)}{\sum_{g=1}^G\sum_{h=1}^Hp(\theta_g,\tilde{\sigma}_h^2,y_1,\cdots,y_n)}
\end{array}
$$

>[!tip] Tips
>- $\theta_k,\tilde{\sigma}_l^2$ 的选取需要等距地选取
>- 计算复杂度为 $M^n$, 其中 $M$ 为样本数量，$n$ 为未知分布参数个数。因此 Discrete Approximation 只适用于参数较少的情况。

## Gibbs Sampler

$$
\begin{array}{l}
1. &\text{Sample }\theta^{(i)}\sim p(\theta|\tilde{\sigma}^{2(i-1)},y_1,\cdots,y_n)\\
2. &\text{Sample }\tilde{\sigma}^{2(i)}\sim p(\tilde{\sigma}^2|\theta^{(i)},y_1,\cdots,y_n)\\
3. &\text{Let }\Phi^{(i)}=\{\theta^{(i)},\tilde{\sigma}^{2(i)}\}
\end{array}
$$