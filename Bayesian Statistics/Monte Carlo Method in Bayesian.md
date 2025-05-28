
>[! Question] Why Monte Carlo Method?
>实际上我们在应用中会更关注于对后验分布的某些方面的总结，如计算一个较为抽象的集合的概率 $\Pr(\theta\in A|y_1,\cdots,y_n)$ for arbitrary set $A$, 或者关注分布的均值与标准差，或者关注于两个或多个样本之间的后验分布的关系，如 $|\theta_1-\theta_2|,\theta_1/\theta_2,\max\{\theta_1,\theta_2,\cdots,\theta_n\}$。针对这些更复杂的情况，我们使用常规方法计算随机变量函数的分布会非常麻烦，因此使用 Monte Carlo Method 生成样本近似估计上述的情况会更加简便。

**Idea:** using random samples and LLN to get numerical values.

We can use MC to do
- Posterior inference on $\theta$
- Posterior inference for arbitrary functions of $\theta$
- Sampling from predictive distributions
- Posterior predictive model checking

$\theta^{(1)},\cdots,\theta^{(s)}\overset{iid}\thicksim p(\theta|y_1,\cdots,y_n)$
- $\frac{1}{s}\sum_i\theta^{(i)}\rightarrow E[\theta|y_1,\cdots,y_n]$
- $\frac{1}{s}\sum_i g(\theta^{(i)})\rightarrow E[g(\theta)|y_1,\cdots,y_n]$
- $\frac{1}{s-1}\sum_i (\theta^{(i)}-\bar \theta)^2\rightarrow Var(\theta|y_1,\cdots,y_n)$
- $\frac{\#\{\theta^{(i)}\leq c\}}{s}\rightarrow \Pr(\theta\leq c)$

# Monte Carlo Standard Errors

Monte Carlo standard error is the approximation to the standard deviation.

An approximate $95\%$ Monte Carlo confidence interval for the posterior mean of $\theta$ is
$$
\hat\theta\pm2SE(\hat\theta)\quad\text{where }SE(\hat\theta)=\frac{\hat\sigma}{\sqrt{S}}=\frac{\sqrt{\frac{1}{S-1}\sum_i(\theta^{(i)}-\bar \theta)^2}}{\sqrt{S}}
$$

# Posterior Inference for Arbitrary Functions

For example, consider log odds:
$$
\log\text{odds}(\theta)=\log\frac{\theta}{1-\theta}=g(\theta)
$$
To get the posterior inference for arbitrary functions $g(\theta)$, we can use Monte Carlo Method to generate samples:

$$
\left.\begin{aligned}

    &\text{sample } \theta^{(1)} \sim p(\theta \mid y_1, \ldots, y_n), \quad \text{compute } \gamma^{(1)} = g(\theta^{(1)}) \\

	&\text{sample } \theta^{(2)} \sim p(\theta \mid y_1, \ldots, y_n), \quad \text{compute } \gamma^{(2)} = g(\theta^{(2)}) \\

    &\vdots \\

    &\text{sample } \theta^{(S)} \sim p(\theta \mid y_1, \ldots, y_n), \quad \text{compute } \gamma^{(S)} = g(\theta^{(S)})

\end{aligned} \right\}
 \text{independently}
$$

Similarly, we can use these samples to get the similar approximations as above.

# Sampling from Predictive Distributions

>[!Example] Poisson Model (Sampling $\tilde Y|y_1,\cdots,y_n\sim NegBinomial$)
>**Step 1**
>$$
>\theta_1^{(i)}\thicksim p(\theta_1|y_{i,1})\quad\theta_2^{(i)}\thicksim p(\theta_2|y_{i,2})
>$$
>**Step 2**
>$$
>\tilde{Y}_1^{(i)}\thicksim Poisson(\theta_1^{(i)})\quad\tilde{Y}_2^{(i)}\thicksim Poisson(\theta_2^{(i)})
>$$
>for each value of $\theta_1^{(i)},\theta_2^{(i)}$.
>
>The sequence $\{(\theta,\tilde y)^{(i)}\}$ constitutes $S$ independent samples from the joint posterior distribution of $(\theta,\tilde y)$, and $\{\tilde y^{(i)}\}$ constitutes $S$ independent samples from the marginal posterior distribution of $\tilde Y$ , which is the posterior predictive distribution.

# Posterior predictive model checking

**Idea:** sample $\tilde{y}$ many times, does it "look like" your original dataset?

If our model is reasonable, two histograms (original dataset, posterior predictive model) should look like same.

If not same, we need to consider
- change prior model
- change sampling model (Poisson $\rightarrow$ Negative Binomial/ Multinomial)