# Generalized Linear Models

Consider $Y_i=$ the number of eggs laid by bird $i$, $X_i=$ Age (in years) of bird $i$, we have
$$
Y_i|\theta(\text{age})\sim Poisson(\theta(\text{age}))
$$
where $i=\{1,\cdots,6\}$ i.e. we have six groups.

We let 
$$
\theta(age)=\exp[\beta_1+\beta_2Age+\beta_3Age^2]=\exp[\boldsymbol{\beta}^T\mathbf{X}]
$$
In general, $\mathbf{X}$ could contain more predictors.

Then, we can get our link function $g(theta(age))=\log[\theta(age)]=\boldsymbol{\beta}^T\mathbf{X}$.

## Prior Distribution

$$
\boldsymbol{\beta}\sim MVN(\mathbf{0},\Sigma)
$$

## Sampling Model

$$
Y_i|\theta(age)\sim Poisson(\exp[\boldsymbol{\beta}^T\mathbf{X}])
$$

## Posterior Distribution

$$
\begin{array}{rl}
p(\boldsymbol{\beta}|\mathbf{y})&\propto p(\boldsymbol{\beta})p(\mathbf{y}|\boldsymbol{\beta})\\
&\propto \exp[-\frac{1}{2}\boldsymbol{\beta}^T\Sigma^{-1}\boldsymbol{\beta}+\sum_iy_i\boldsymbol{\beta}^T\mathbf{x}_i]\times\exp[-\sum_i\exp(\boldsymbol{\beta}^T\mathbf{x}_i)]
\end{array}
$$
The second term is too complex, we cannot compute even full conditional nor the posterior.

We can do [[Normal Models#Discrete Approximation|grid approximation]] of $p(\boldsymbol{\beta}_1,\boldsymbol{\beta}_2,\boldsymbol{\beta}_3|\mathbf{y},\mathbf{X})$.

## Metropolis Algorithm (MCMC)

We need to approximate
$$
p(\theta|y)=\frac{p(\theta)p(y|\theta)}{\int p(y|\tilde\theta)p(\tilde\theta)d\tilde\theta}
$$
The most difficulty is how to calculate the denominator ${\int p(y|\tilde\theta)p(\tilde\theta)d\tilde\theta}$.

Assume that we have samples $\theta^{(1)},\cdots,\theta^{(s)}$, we want to add $\theta^{(s+1)}$. What should it be?

**Idea:** Propose a new value $\theta^{*}$ near to $\theta^{(s)}$. Make a decision: include or not.
- If $p(\theta^{*}|\mathbf{y})>p(\theta^{(s)}|\mathbf{y})$, then include $\theta^{(s+1)}=\theta^{*}$. $\theta^{*}$ is better than $\theta^{(s)}$.
- If $p(\theta^{*}|\mathbf{y})<p(\theta^{(s)}|\mathbf{y})$, then **maybe** keep it.

Using the trick [[Some Mathematical Ideas in Bayesian Statistics#7. odds 的引入|odds]], we take a ratio
$$
r=\frac{p(\theta^{*}|\mathbf{y})}{p(\theta^{(s)}|\mathbf{y})}=\frac{p(\theta^{*})}{p(\theta^{(s)})}\times\frac{p(\mathbf{y}|\theta^{*})}{p(\mathbf{y}|\theta^{(s)})}
$$
This algorithm is [[Random Variable Generation#Acceptance-Rejection Method|Acceptance-Rejection Method]].

**Metropolis Algorithm:**
1. Sample $\theta^{*}$ near $\theta^{(s)}$
	near: consider proposal distribution $J(\theta^{*}|\theta^{(s)})$. 
	- the proposal distribution is symmetric, i.e. $J(\theta^{*}|\theta^{(s)})=J(\theta^{(s)}|\theta^{*})$
	- the distribution is usually simple, such as $\text{uniform}(\theta^{(s)}-{\sigma},\theta^{(s)}+{\sigma}),N(\theta^{(s)},\sigma^2)$
	- $\sigma$ should be small enough so that $\theta^*$ will not jump.
2. Compute $r=\frac{p(\theta^{*}|\mathbf{y})}{p(\theta^{(s)}|\mathbf{y})}=\frac{p(\theta^{*})}{p(\theta^{(s)})}\times\frac{p(\mathbf{y}|\theta^{*})}{p(\mathbf{y}|\theta^{(s)})}$
3. Make a decision
$$
\theta^{(s+1)}=\begin{cases}
\theta^*,\text{ with prob. }\min(r,1)\\
\theta^{(s)},\text{ with prob. }1-\min(r,1)
\end{cases}
$$
	if $r$ is too small, we can also consider $\log r$ to avoid compute small number, i.e.
	$$
	u\sim \text{uniform}[0,1]\quad \theta^{(s+1)}=\begin{cases}
\theta^*,\text{if }\log r>\log u\\
\theta^{(s)},\text{if }\log r<\log u
\end{cases} 
$$

>[!tips] Run Effectively
>
>What $\sigma$ we choose will have a significant impact on our results. If we choose proper $\sigma$, then we can reduce "burn-in" period and avoid "stuck".
>![[how to choose sigma.png]]
>
>**How to choose $\sigma$?**
>- Large $\sigma\Rightarrow$ $\theta^*$ rejected more often ("stuck") $\Rightarrow$ keep $\theta^{(s)}\Rightarrow$ high ACF
>- Small $\sigma\Rightarrow$ $\theta^{*}$ needs a large number of iterations for MC to move from the starting value of $0$ to posterior mode.
>
>![[ACF of sigma.jpg]]
>
>**Rule of Thumb:**
>
>Run several chains with different $\sigma$ and choose $\sigma$ s.t. $AR(\text{Acceptance Ratio})\in[20\%,50\%]$

$\theta^{(0)}=\hat{\theta}_{MLE}$ is a good choice.

For Poisson regression, we can let
$$
\hat\sigma^2=\text{sample variance of }\begin{cases}
\{\log(Y_i+\frac{1}{2})\}\\
\{\log (Y_i+1)\}\\
\{\log(Y_i+\frac{1}{n})\}
\end{cases}
$$
to avoid $\log(0)$. 

## Metropolis-Hastings Algorithm

This algorithm is [[Random Variable Generation#Acceptance-Rejection Method|Acceptance-Rejection Method]].

Consider $p_0(u,v)=p(\theta,\sigma^2|\mathbf{y})$

4. update $U$:
	- sample $u^*\sim J_u(u|u^{(s)},v^{(s)})$
	- compute the acceptance ratio
	$$
	r=\frac{p_0(u^*,v^{(s)})}{p_0(u^{(s)},v^{(s)})}\times\frac{J_u(u^{(s)}|u^*,v^{(s)})}{J_u(u^*|u^{(s)},v^{(s)})}
	$$
	- set $$u^{(s+1)}=\begin{cases}u^*,\text{with prob. }\min(1,r)\\ u^{(s)},\text{with prob. }\max(0,1-r)\end{cases}$$
5. update $V$:
	- sample $v^*\sim J_v(v|u^{(s+1)},v^{(s)})$
	- compute the acceptance ratio
	$$
		r=\frac{p_0(u^{(s+1)},v^{*})}{p_0(u^{(s+1)},v^{(s)})}\times\frac{J_v(v^{(s)}|u^{(s+1)},v^{*})}{J_v(v^*|u^{(s+1)},v^{(s)})}
	
	$$
	- set $$v^{(s+1)}=\begin{cases}v^*,\text{with prob. }\min(1,r)\\ v^{(s)},\text{with prob. }\max(0,1-r)\end{cases}$$

From the formula for the acceptance ratio, we can find that
$$
r=\text{Metropolis part}\times\text{Correlation Factor}
$$
If a value $u^*$ is much more likely to be proposed than the current value $u^{(s)}$, then we must **down-weight the probability of accepting $u^*$ accordingly**, otherwise the value $u^*$ will be overrepresented in our sequence.(目的就是防止重复采样，这样会导致我们采样的效率极低，类似于[[Frequency Statistics#Adjusted $R 2$ $R 2_{adj,p}$|Adjusted R square]]添加一项 penalty term, see [[Model Selection and Statistical Information#Regularization|Regularization]])

- The proposal distributions $J_u,J_v$ are not required to be symmetric. The only requirement is that they do not depend on $U$ or $V$ values in our sequence previous to the most current values. This ensures that the sequence is a Markov Chain.
- If the proposal distributions are symmetric, then the algorithm will become Metropolis.
- If $J_u(u^*|u^{(s)},v^{(s)})=p_0(u^*|v^{(s)})$, the algorithm will become Gibbs.

>[!Question] Why does Metropolis-Hasting Algorithm Work?
>在本书中，Metropolis-Hasting Algorithm 生成的马尔可夫链都是 irreducible, aperiodic and recurrent。我们只需要证明平稳分布 $\Pr(x^{(s+1)}=x)=p_0(x)$，因为我们使用 Metropolis-Hasting Algorithm 的目标就是利用 mcmc 去模拟后验分布。
>
>**Proof:**
>Let $x_a$ and $x_b$ be any two values of $X$ such that $p_0(x_a)J_s(x_b|x_a)\geq p_0(x_b)J_s(x_a|x_b)$. Then we can get the probability of $x^{(s)}=x_a,x^{(s+1)}=x_b$
>$$\begin{array}{rl}
>\Pr(x^{(s)}=x_a,x^{(s+1)}=x_b)&= p_0(x_a)\times J_s(x_b|x_a)\times \frac{p_0(x_b)J_s(x_a|x_b)}{p_0(x_a)J_s(x_b|x_a)}\\
>&=p_0(x_b)J_s(x_a|x_b)
>\end{array}$$
>Similarly, we can get the probability of $x^{(s)}=x_b,x^{(s+1)}=x_a$
>$$
>\begin{array}{rl}
>\Pr(x^{(s)}=x_b,x^{(s+1)}=x_a)&= p_0(x_b)\times J_s(x_a|x_b)\times \frac{p_0(x_a)J_s(x_b|x_a)}{p_0(x_b)J_s(x_a|x_b)}\\
>&=p_0(x_b)J_s(x_a|x_b)\times 1\\
>&=p_0(x_b)J_s(x_a|x_b)
>\end{array}
>$$
>i.e. $\Pr(x^{(s)}=x_b,x^{(s+1)}=x_a)=\Pr(x^{(s)}=x_a,x^{(s+1)}=x_b)$ for any two values $x_a$ and $x_b$. Therefore, we can get
>$$
>\begin{array}{rl}
> \Pr(x^{(s+1)}=x)&=\sum_{x_{a}}\Pr(x^{(s+1)}=x,x^{(s)}=x_{a})\\ &=\sum_{x_{a}}Pr(x^{(s+1)}=x_{a},x^{(s)}=x)\\
&=\Pr(x^{(s)}=x)\\
\end{array}
>$$

# Regression Model with Correlated Errors

Consider $\mathbf{Y}=\begin{pmatrix}Y_{1}\\\vdots\\Y_n\end{pmatrix}\sim MVN(\mathbf{X}\boldsymbol{\beta},\Sigma)$ where
$$
\Sigma=\sigma^2\mathbf{C}_\rho=\sigma^2\begin{pmatrix}
1&\rho&\rho^2&\cdots&\rho^{n-1}\\
\rho&1&\rho&\cdots&\rho^{n-2}\\
\rho^2&\rho&1&&&\\
\vdots&\vdots&&\ddots&\\
\rho^{n-1}&\rho^{n-2}&&&1
\end{pmatrix}
$$

## Prior

We use the same [[Linear Regression#Semi-conjugate Prior|semi-conjugate prior]] of $\boldsymbol{\beta},\sigma^2$.  

## Posterior

$$
\begin{array}{c}
\{\boldsymbol{\beta}|\mathbf{X},\mathbf{y},\sigma^2,\rho\}\sim MVN(\boldsymbol{\beta}_n,\Sigma_n),\text{where}\\
\Sigma_n=(\mathbf{X}^T\mathbf{C}_\rho^{-1}\mathbf{X}/\sigma^2+\Sigma_0^{-1})^{-1}\\
\boldsymbol{\beta}_n=\Sigma_n(\mathbf{X}^T\mathbf{C}_\rho^{-1}\mathbf{y}/\sigma^2+\Sigma_0^{-1}\boldsymbol{\beta}_0)
\end{array}
$$
$$
\begin{array}{c}
\{\sigma^2 | \mathbf{X}, \mathbf{y}, \boldsymbol{\beta}, \rho \} \sim \text{inverse-gamma}\left(\frac{v_0 + n}{2}, \frac{v_0\sigma_0^2 + \text{SSR}_\rho}{2}\right), \, \text{where}\\
\text{SSR}_\rho = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T\mathbf{C}_\rho^{-1}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
\end{array}
$$

If $\beta_0=0,\Sigma_0\rightarrow \infty$, i.e. flat prior
- $\Sigma_n\rightarrow \sigma^2(\mathbf{X}^T\mathbf{C}_\rho^{-1}\mathbf{X})^{-1}$
- $\boldsymbol{\beta}_n\rightarrow (\mathbf{X}^T\mathbf{C}_\rho^{-1}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{C}_\rho^{-1}\mathbf{y}$ (Generalized Least Squares estimate of $\boldsymbol{\beta}$)

## Metropolis + Gibbs

1. Update ($\boldsymbol{\beta}$): Sample $\boldsymbol{\beta}^{(s+1)} \sim \text{multivariate normal}(\boldsymbol{\beta}_n, \Sigma_n)$, where $\boldsymbol{\beta}_n$ and $\Sigma_n$ depend on $\sigma^{2(s)}$ and $\rho^{(s)}$.

2. Update $\sigma^2$: Sample $\sigma^{2(s+1)} \sim \text{inverse-gamma}\left(\frac{\nu_0 + n}{2}, \frac{\nu_0\sigma_0^2 + \text{SSR}_\rho}{2}\right)$,  where $\text{SSR}_\rho$ depends on $\boldsymbol{\beta}^{(s+1)}$ and $\rho^{(s)}$.

3. Update $\rho$:
	- Propose $\rho^* \sim \text{uniform}(\rho^{(s)} - \delta, \rho^{(s)} + \delta)$
		- If $\rho^* < 0$, then reassign it to be $|\rho^*|$. 
		- If $\rho^* > 1$, reassign it to be $2 - \rho^*$.
	- Compute the acceptance ratio $$r = \frac{p(\boldsymbol{y}|\mathbf{X}, \boldsymbol{\beta}^{(s+1)}, \sigma^{2(s+1)}, \rho^*)p(\rho^*)}{p(\boldsymbol{y}|\mathbf{X}, \boldsymbol{\beta}^{(s+1)}, \sigma^{2(s+1)}, \rho^{(s)})p(\rho^{(s)})}$$and sample $u \sim \text{uniform}(0,1)$. 
		- If $u < r$, set $\rho^{(s+1)} = \rho^*$
		- otherwise set $\rho^{(s+1)} = \rho(s)$.

The proposal distribution used in Step 3.1 is called a **reflecting random walk**, which ensures that $0<\rho<1$.
