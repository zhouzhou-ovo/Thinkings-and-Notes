# Frequentist coverage(Pre-experiment Coverage)

$(1-\alpha)100\%$ CI is defined as a **random interval** $(l(Y),u(Y))$, where
$$
\Pr(l(Y)<\theta<u(Y)|\theta)=1-\alpha
$$

# Bayesian CI/ Confidence Region(Post-experiment Coverage)

$(l(Y),u(Y))$**(fixed)** is a $(1-\alpha)100\%$ confidence region if
$$
\Pr(l(Y)<\theta<u(Y)|Y=y)=1-\alpha
$$

# Highest Posterior Density Interval(HPD)

$(1-\alpha)100\%$ **Highest Posterior Density Region** is a confidence region $S(y)$ such that
*1.* $\Pr(\theta\in S(y)|Y=y)=1-\alpha$
*2.* $\forall \theta_a\in S(y)$ and $\forall \theta_b \notin S(y)$
$$
\Pr(\theta_a|Y=y)>\Pr(\theta_b|Y=y)
$$

>[!Question] How to get HPD?
>**Ans:**
>Gradually move a horizontal line down across the density, including in the HPD region all $\theta$-values having a density above the horizontal line. Stop moving the line down when the posterior probability of the $\theta$-values in the region reaches $(1-\alpha)$. 

For binomial model(asymmetric), HPD is narrower than Quantile-based interval.
- For symmetric model, HPD $=$ QCI
- For asymmetric model, HPD is narrower than QCI

# Quantile Based Interval

$(1-\alpha)100\%$ Quantile based interval is an interval of the form
$$
(\theta_{\alpha/2},\theta_{1-\alpha/2})
$$
