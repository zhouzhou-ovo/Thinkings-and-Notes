 
# Posterior Approximation with the Gibbs Sampler

**Idea:** Joint posterior distribution is non-standard and difficult to sample from directly. Sampling from the **full conditional distribution** of each parameter is easy! E.g. Normal Model with semi-conjugate prior.

## General Setup of the Gibbs Sampler

We consider $\Phi=\{\phi_1,\cdots,\phi_p\}$. Uncertainty in $\Phi$ is captured by $p(\Phi)=p(\phi_1,\cdots,\phi_p)$.

Given starting values $\Phi^{(0)}=\{\phi_1^{(0)},\cdots,\phi_p^{(0)}\}$.
$$\left.
\begin{array}{rl}
1.&\text{sample }\phi_1^{(i)}\sim p(\phi_1|\phi_2^{(i-1)},\phi_3^{(i-1)},\cdots,\phi_p^{(i-1)})\\
2.&\text{sample }\phi_2^{(i)}\sim p(\phi_2|\phi_1^{(i)},\phi_3^{(i-1)},\cdots,\phi_p^{(i-1)})\\
3.&\text{sample }\phi_3^{(i)}\sim p(\phi_3|\phi_1^{(i)},\phi_2^{(i)},\cdots,\phi_p^{(i-1)})\\
\vdots\\
\text{p}.&\text{sample }\phi_p^{(i)}\sim p(\phi_2|\phi_1^{(i)},\phi_2^{(i)},\cdots,\phi_{p-1}^{(i)})
\end{array}\right\}1\text{ scan of the Gibbs Sampler}
$$

Then we repeat for S scan and get a sequence:
$$
\left.
\begin{array}{c}
\Phi^{(1)}=\{\phi_1^{(1)},\cdots,\phi_p^{(1)}\}\\
\Phi^{(2)}=\{\phi_1^{(2)},\cdots,\phi_p^{(2)}\}\\
\vdots\\
\Phi^{(S)}=\{\phi_1^{(S)},\cdots,\phi_p^{(S)}\}
\end{array}
\right\}S\text{ scan of the Gibbs Sampler}
$$
And we get a Markov Chain.

![[Gibbs Sampler Plot.jpg]]

>[!Note] Gibbs Sampler 的特点
>通过 Gibbs Sampler 生成样本点的过程，我们可以发现，参数的第 i 个样本的生成需要依赖于其他已生成的其他参数的第 i 个样本，这说明参数之间并不是类似于 Monte Carlo Method 的互相独立的生成样本，而是相关的。可以联想一个个互相咬合的齿轮，后者的启动需要前面齿轮的带动！


|            |                             Markov Chain Monte Carlo                              |                                                     Monte Carlo                                                     |
| :--------: | :-------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
|   Sample   |                        $\frac{1}{S}\sum_{i=1}^S\phi^{(i)}$                        |                                         $\frac{1}{S}\sum_{i=1}^S\phi^{(i)}$                                         |
|  Basement  |                                   Markov Chains                                   |                                              Central Limit of Theorem                                               |
| Diagnostic | autocorrelation $\Rightarrow$ "stuck" $\Rightarrow$ acf and Seff <br>stationarity | empirical distribution, [[Monte Carlo Method in Bayesian#Monte Carlo Standard Errors\|Monte Carlo Standard Errors]] |

# Introduction to MCMC Diagnostics

Sometimes if we have fewer iterations, the values of parameter get "stuck" in certain regions, i.e. **autocorrelation**.

Gibbs sampler is guaranteed to be eventually stationary, but "eventually" can be a very long time in some situations.

## Stationarity

We want to see the particle has a chance to move out of lower probability regions and into higher probability regions. This is called that the chain has achieved stationarity or has converged.

Non-stationarity is easier to detect than stationarity.

We can use Gelman-Rubin statistic to determine whether the Markov Chain has converged. It compares variation within chains' with variation between chains'. (like ANOVA) The rule of thumb is if the statistics $<1.1$, then we consider the chain has converged.

## Autocorrelation and Effective Sample Size

We can prove that
$$
\begin{array}{rl}
Var_{MCMC}[\bar\phi]&=E[(\bar\phi-\phi_0)^2]\\
&=Var_{MC}[\bar\phi]+\frac{1}{S^2}E[(\phi^{(s)}-\phi_0)(\phi^{(t)}-\phi_0)]
\end{array}
$$
where $\bar\phi=\frac{\sum_s \phi^{(s)}}{S},\phi_0=\int\phi p(\phi)\phi$. From the formula above, we can know that the MCMC variance is usually higher than the MC variance. The higher teh autocorrelation in the chain, the larger the MCMC variance and the worse the approximation is.

We define the sample autocorrelation function to assess how much correlation there is
$$
\text{acf}_t(\boldsymbol{\phi})=\frac{\frac{1}{S-t}\sum_{s=1}^{S-t}(\phi_s-\bar\phi)(\phi_{s+t}-\bar \phi)}{\frac{1}{S-1}\sum_{s=1}^S(\phi_s-\bar \phi)^2}
$$
The higher the autocorrelation, the more MCMC samples we need to attain a given level of precision for our approximation.

We can use effective sample size to measure this. The effective sample size function estimates the value $S_{\text{eff}}$ such that
$$
Var_{MCMC}[\bar\phi]=\frac{Var[\phi]}{S_{\text{eff}}}
$$
$S_{\text{eff}}$ can be interpreted as the number of independent Monte Carlo samples necessary to give the same precision as the MCMC samples.