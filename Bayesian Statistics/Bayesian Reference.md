
Bayesian Inference: updating beliefs in light of new information

Bayesian Methods
- parameter estimates with good statistical properties
- parsimonious descriptions of observed data
- predictions for missing data and forecasts of future data
- a computational framework for model estimation, selection and validation

Limitation of Frequency Approach

If the sample size is too small, the result of inference may not be good.

>[!Example] Infection of a City ('A First Course in Bayesian', Page $7$)
>We use Frequency Approach to get the Wald interval, given by
>$$
>\bar y \pm 1.96\sqrt{\bar y(1-\bar y)/n}
>$$
>If $n$ is large enough, then this interval will be useful; but if $n$ is small, such as $n=20$, the probability that the interval contains parameter $\theta$ is only about $80\%$.
>
>At the same time, for sample in which $\bar y=0$ the Wald CI comes out to be just a single point $0$. In fact, the $99.99\%$ Wald intercal also comes out to be zero. We would not want to conclude from the survey that we are $99.99\%$ certain that on one in the city is infected.

---
# Some Basic Notations

sample space: $\mathcal{Y}$      parameter space: $\Theta$  
- Prior distribution $\pi(\theta)$ or $p(\theta)$: measures the belief that $\theta$ represents true value of the parameter
- Sampling model $p(y|\theta)$: describes our belief that $y$ is the outcome if we know the value $\theta$
- Posterior distribution $p(\theta|y)$: measures the belief that $\theta$ is a true parameter given observed $y$
- Bayes' Theorem:
$$
p(\theta|y)=\frac{p(y|\theta)p(\theta)}{p(y)}=\frac{p(y|\theta)p(\theta)}{\int_{\Theta}p(y|\tilde{\theta})p(\tilde{\theta})d\tilde{\theta}}
$$

**Belief Functions**

>[!Note] Definition (Belief Function)
>Let $\text{Be}()$ be a **belief function**, that is, a function that assigns numbers to statements such that the larger the number, the higher degree of belief.
- $\text{Be}(F)>\text{Be}(H)$ means we would prefer to bet F is true than G is true
- $\text{Be}(F|H)>\text{Be}(G|H)$ means that if we knew that H were true, then we would prefer to bet that F is also true than bet G is also true
- $\text{Be}(F|G)>\text{Be}(F|H)$ means that if we were forced to bet on F, we would prefer to do it under the condition that G is true rather than H is true
- Axioms of beliefs and probability
	- **B1$\Leftrightarrow$P1** $\text{Be}(\text{not}\,H|H)\leq \text{Be}(F|H)\leq \text{Be}(H|H)$ $\Leftrightarrow$ $0=\Pr(\text{not}\,H|H)\leq \Pr(F|H)\leq\Pr(H|H)=1$
	- **B2$\Leftrightarrow$P2** $\text{Be}(F\,\text{or}\,G|H)\geq\max\{\text{Be}(F|H),\text{Be}(G|H)\}$ $\Leftrightarrow$ $\Pr(F\cup G|H)=\Pr(F|H)+\Pr(G|H)\,\text{if}\,F\cap G=\emptyset$
	- **B3$\Leftrightarrow$P3** $\text{Be}(F\,\text{and}\,G|H)$ can be derived from $\text{Be}(G|H)$ and $\text{Be}(F|G\,\text{and}\,H)$ $\Leftrightarrow$ $\Pr(F\cap G|H)=\Pr(G|H)\Pr(F|G\cap H)$ 
实际上就是基于条件概率的乘法公式进行不断的变换

**Exchangeability**

>[!Note] Definition (Exchangeable)
>Let $p(y_1,\cdots,y_n)$ be the joint density of $Y_1,\cdots,Y_n$. If $p(y_1,\cdots,y_n)=p(y_{\pi_1},\cdots,y_{\pi_n})$ for all permutations $\pi$ of $\{1,\cdots,n\}$, then $Y_1,\cdots,Y_n$ are **exchangeable**.

Roughly speaking, $Y_1,\cdots,Y_n$ are exchangeable if the subscript labels convey no information about the outcomes.

>[!Note] Claim
>If $\theta\thicksim p(\theta)$ and $Y_1,\cdots,Y_n$ are conditionally i.i.d. given $\theta$, then marginally $Y_1,\cdots,Y_n$ are exchangeable.
>**Proof:**
>$$
>\begin{array}{rl}
>p(y_1,\cdots,y_n)&=\int p(y_1,\cdots,y_n|\theta)p(\theta)d\theta\\
>&=\int \left\{\prod_{i=1}^np(y_i|\theta)\right\}p(\theta)d\theta\\
>&=\int \left\{\prod_{i=1}^np(y_{\pi_i}|\theta)\right\}p(\theta)d\theta\\
>&= p(y_{\pi_1},\cdots,y_{\pi_n})
>\end{array}
>$$

>[!Note] Theorem (de Finetti)
>Let $Y_i\in \mathcal{Y}$ for all $i\in\{1,2,\cdots\}$. Suppose that, for any $n$, our belief model for $Y_1,\cdots,Y_n$ is exchangeable:
>$$
>p(y_1,\cdots,y_n)=p(y_{\pi_1},\cdots,y_{\pi_n})
>$$
>for all permutations $\pi$ of $\{1,\cdots,n\}$. Then our model can be written as
>$$
>p(y_1,\cdots,y_n)=\int \left\{\prod_{i=1}^np(y_i|\theta)\right\}p(\theta)d\theta
>$$
>for some parameter $\theta$, some prior distribution on $\theta$ and some sampling model $p(y|\theta)$. The prior and sampling model depend on the form of the belief model $p(y_1,\cdots,y_n)$, i.e.
>$$
>\begin{aligned}
& Y_1, \ldots, Y_n \mid \theta \text{ are i.i.d.} \\
& \theta \sim p(\theta)
\end{aligned}
\bigg\}
\iff Y_1, \ldots, Y_n \text{ are exchangeable for all } n.
>$$








