#MonteCarlo 
All random numbers generated from $U[a,b]$ base on [[Random Number Generation]].
# Inverse-Transform Method

Some ideas can see in [[Some practical conclusions and mathematical ideas in Introduction to Data Science#1. The distrib. of CDF as a RV|The Distribution of CDF as a Random Variable]].

Let $X$ be a random variable with c.d.f. $F$, which is a non-decreasing function, then we have
$$
F^{-1}(y)=\inf\{x:F(x)\geq y\},\quad 0\leq y \leq 1
$$
We can show that if $U\sim U[0,1]$, then
$$
X=F^{-1}(U)
$$
i.e.
$$
\mathbb{P}(X\leq x)=\mathbb{P}(F^{-1}(U)\leq x)=\mathbb{P}(U\leq F(x))=F(x)
$$
We have the algorithm as below

![[inverse transform method algorithm.png]]

>[! example] Drawing from a Discrete Distribution
>![[it method for discrete rv.png]]
>![[it method for discrete rv algorithm.png]]

>[! warning] When can use inverse-transform method?
>In general, the inverse-transform method  requires that the underlying **c.d.f., $F$, exist in a form** for which the corresponding **inverse function $F^{−1}$** can be found analytically or algorithmically. 
>
>For many other probability distributions, it is either impossible or difficult to find the inverse transform, that is, to solve
>$$
>F(x)=\int_{-\infty}^xf(t)dt=u
>$$
>w.r.t. $x$. Therefore, the inverse-transform method may not be effcient.

# Alias Method

**Idea:** An arbitrary **discrete** $n$-point p.d.f. $f$, with
$$
f(x_i)=\mathbb{P}(X=x_i),\quad i=1,\cdots,n
$$
can be represented as an **equally weighted** mixture of $n$ p.d.f., $q^{(k)},k=1,\cdots,n$, each having at most two nonzero components, that is
$$
f(x)=\frac{1}{n}\sum_{k=1}^nq^{(k)}(x)
$$
![[alias method algorithm.png]]

Alias method typically use $O(n\log n)$ or $O(n)$ preprocessing time, after which random values can be drawn from the distribution in $O(1)$ time.

# Composition Method

**Idea:** A c.d.f. $F$ can be expressed as a mixture of c.d.f. $\{G_i\}$, that is
$$
F(x)=\sum_{i=1}^mp_iG_i(x)
$$
where $p_i>0,\,\sum_{i=1}^mp_i=1$.

Let $X_i\sim G_i$ and let $Y$ be a discrete random variable with $\mathbb{P}(Y=i)=p_i$, independent of $X_i$, for $1\leq i\leq m$. Then a random variable $X$ with c.d.f. $F$ can be represented as
$$
X=\sum_{i=1}^mX_iI_{\{Y=i\}}
$$
We only need to generate $X_i$.
![[composition method algorithm.png]]

>[!note] Alias Method v.s. Composition Method
>- Same: 两种方法都是基于对目标分布函数进行分解，表示为多个子分布的加权和
>- Different:
>	- Alias method 子分布的权重相同，只适用于离散分布
>	- Composition method 子分布的权重需要自己生成，适用于混合分布或者组合分布，在文中的情况下，alias 应该是 composition 的一种特例形式。

>[!note] Monte Carlo Method 的部分分布到总体分布
>在上文的 Composition Method 的算法部分，我们发现，在生成目标分布的随机数时，我们并不需要使用所有的子分布参与到模拟中，我们只需要提取一个子分布去生成这个分布下的随机数，即
>$$
>\text{random number of sub-distribution} = \text{random number of target distribution}
>$$
>核心就是**目标分布与分解的子分布的加权和的一一对应**。

# Acceptance-Rejection Method

**Idea:** Using the boundedness of p.d.f., generate uniform random numbers to divide p.d.f. into two regions: acceptance and rejection.

Suppose that the target p.d.f $f$ is bounded on some finite interval $[a,b]$ and is zero outside this interval. Let
$$
c=\sup\{f(x):x\in[a,b]\}
$$
![[acceptance-rejection method plot.png]]

Then we can generate $Z\sim f$ straightly as
1. Generate $X\sim U(a,b)$
2. Generate $Y\sim U(0,c)$ independently of $X$
3. If $Y\leq f(X)$, return $Z=X$. Otherwise, return to Step 1.

We generalize this procedure as follows: Let $g$ be an arbitrary density s.t. $\phi(x)=Cg(x)$ majorizes $f(x)$ for some constant $C$, i.e. $\phi(x)\geq f(x)$ for all $x$, where $C$ is necessarily larger than $1$. We call **$g(x)$ the proposal p.d.f.**.
![[generalize ar methods.png]]

![[generalize ar method algorithm.png]]

>[!question] Why our criterion is if $Y\leq f(x)$ return $Z=f(x)$?
>Our criterion is to accept all points **above** the uniform random number as the threshold and reject those below. This criterion ensures that we can generate **all points in $[a,b]$** that follows $f(x)$. If we accept those below the threshold, it can not ensure us to get all points in $[a,b]$.

## Feasibility of Acceptance-Rejection Algorithm

We also have a theorem that supports our algorithm.

**Theorem:** The random variable generated according to Algorithm $2.3.5$ has the desired p.d.f. $f(x)$.
	*Proof:* We define two subsets$$
	A=\{(x,y):0\leq y\leq Cg(x)\}\quad B=\{(x,y):0\leq y\leq f(x)\}
	$$
	Then we consider the joint p.d.f. $q(x,y)$ of $(X,Y)$ and let $q(y|x)$ denote the conditional p.d.f of $Y$ given $X=x$. Then we have$$
	q(x,y)=\begin{cases}g(x)q(y|x)\quad \text{if}(x,y)\in A\\0\quad\text{otherwise}\end{cases}
	$$
	For $q(y|x)=1/(Cg(x)),y\in [0,C(g(x))]$, then we have $q(x,y)=C^{-1},(x,y)\in A$ **uniformly**. Therefore, for accepted point $(X^*,Y^*)\in B$ with $\int f(x)=1$, we can get the marginal p.d.f of $X$ as $$
	\int_{0}^{f(x)}1dy=f(x)
	$$
The efficiency of Algorithm $2.3.5$ is defined as
$$
\mathbb{P}((X,Y)\text{ is accepted})=\frac{\text{area }A}{\text{area }B}=\frac{1}{C}
$$
We also call it **acceptance ratio**.

If we use acceptance ration in our algorithm, we have 
![[modified ar method algorithm.png]]

**Inference**
Actually, we notice that $Y=UCg(x),U\sim U(0,1)$ and
$$
p=\mathbb{P}(U\le\frac{f(x)}{Cg(x)})=\mathbb{P}(Y\le f(X))=\frac{1}{C}
$$
for each $(X,U)$. Since the trials are independent, the number of trials, $N$, before a successful pair $(Z,U)$ occurs has the following geometric distribution
$$
\mathbb{P}(N=n)=p(1-p)^{n-1},\quad n=1,\cdots
$$

## Criteria

1. It should be easy to generate a random variable from $g(x)$.
2. The efficiency, $1/C$, of the procedure should be large; that is, $C$ should be close to $1$ (which occurs when $g(x)$ is close to $f (x)$).

[[Generalized Linear Models#Metropolis Algorithm (MCMC)|Metropolis Algorithm]] and [[Generalized Linear Models#Metropolis Algorithm (MCMC)|Metropolis-Hasting Algorithm]]Metropolis-Hasting Algorithm are two kinds of acceptance-rejection method.

# Generating Continuous Random Variable

## Normal (Gaussian) Distribution

We all generate standard normal distribution in this part.
### Box-Muller Approach

Consider $(X,Y)$ are two independent standard normal random variables. Then we consider their corresponding polar coordinates $(R,\Theta)$
$$
f_{R,\Theta}(r,\theta)=\frac{1}{2\pi}e^{-r^2/2}r,\quad r\ge0,\theta\in[0,2\pi)
$$
where $x=r\cos\theta,y=r\sin\theta$.
We can get this joint p.d.f. by [[Probability Reference#General Transformation for Random Variables|General Transformation for Random Variables]].

We take integral w.r.t. $R,\Theta$ respectively
$$
\Theta\sim U[0,2\pi)\quad \mathbb{P}(R>r)=e^{-r^2/2}
$$
Using [[Random Variable Generation#Inverse-Transform Method|Inverse-Transform Method]], we can get $X,Y$ independently.

![[Box-Muller Algorithm.png]]

### Acceptance-Rejection Method

By symmetry of standard normal distribution, we can consider Standard Half-Normal Distribution $X$ that
$$
f(x)=\sqrt{\frac{2}{\pi}}e^{-x^2/2},x\ge 0
$$
We consider proposal distribution $g(x)=e^{-x}$ and $C=\sqrt{2e/\pi}$ such that
$$
f(x)\le Cg(x)
$$
The efficiency of this method is therefore $\sqrt{\pi/2e}\approx 0.76$. And the acceptance condition is
$$
U\le f(X)/(Ce^{-X})\Rightarrow -\ln U\geq \frac{(X-1)^2}{2}
$$
where $V_1=-\ln U\sim Exp(1)$ and $X\sim Exp(1)$. (Because $g(x)=e^{-x}$)

