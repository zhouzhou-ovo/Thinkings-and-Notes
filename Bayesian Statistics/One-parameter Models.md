 
# Exponential Families

>[!Note] Definition (Exponential Family Model)
>
>A one-parameter **exponential family model** is any model whose densities can be expressed as $p(y|\Phi)$
>$$
>p(y|\Phi)=h(y)c(\Phi)\exp\{\Phi t(y)\}
>$$
>where $\Phi$ is a parameter, and $t(y)$ is a [[Some Mathematical Ideas in Bayesian Statistics#4. 充分统计量（Sufficient Statistics）|sufficient statistic]].

## Conjugate Prior Distribution and Posterior Distribution

$$
p(\Phi|n_0,t_0)=k(n_0,t_0)c(\Phi)^{n_0}\exp(n_0t_0\Phi)
$$
We can calculate that
$$
\begin{array}{rl}
p(\Phi|y_1,\cdots,y_n)\propto& p(\Phi)p(y_1,\cdots,y_n|\Phi)\\
\propto& c(\Phi)^{(n_0+n)}\exp[(n_0+n)\frac{n_0t_0+nt}{n_0+n}\Phi]\\
\end{array}
$$
where $t=\sum t(y_i)$.
We can get posterior sufficient statistics is  weight of prior sufficient statistics and data sufficient statistics.

For $t_0$, we can get more precise:
$$
E[t(Y)]=E[E[t(Y)|\phi]]=E[-c'(\phi)/c(\phi)]=t_0
$$
**Proof:**
	First, we prove $E[t(Y)|\phi]=-c'(\phi)/c(\phi)$.$$
	\begin{array}{c}
	\int p(y|\phi)dy=1\\
	\Leftrightarrow \int h(y)c(\phi)\exp[\phi t(y)]dy=1\\
	\Leftrightarrow \int h(y)\{c'(\phi)\exp[\phi t(y)]
	+t(y)c(\phi)\exp[\phi t(y)]\}dy=0\\
	(\text{Take derivatives w.r.t to }\phi)\\
	\Leftrightarrow \frac{c'(\phi)}{c(\phi)}+E[t(y)|\phi]=0\\
	\Leftrightarrow E[t(y)|\phi]=-\frac{c'(\phi)}{c(\phi)}
	\end{array}$$
	Second, we prove that $E[-c'(\phi)/c(\phi)]=t_0$.$$
	\begin{array}{c}
	p(\phi|n_0,t_0)=\kappa_0c(\phi)^{n_0}\exp(n_0t_0\phi)\\
	\Leftrightarrow \frac{dp(\phi|n_0,t_0)}{d\phi}=\kappa_0n_0c'(\phi)c(\phi)^{n_0-1}\exp(n_0t_0\phi)\\+\kappa_0n_0t_0c(\phi)^{n_0}\exp(n_0t_0\phi)\\
	\Leftrightarrow 0=n_0\int_\Phi\frac{c'(\phi)}{c(\phi)}\kappa_0c(\phi)^{n_0-1}\exp(n_0t_0\phi)d\phi\\+n_0t_0\int_\Phi\kappa_0n_0t_0c(\phi)^{n_0}\exp(n_0t_0\phi)d\phi\\
	\Leftrightarrow E[-c'(\phi)/c(\phi)]=t_0
	\end{array}$$

---
# Binomial Model

Example can be seen in [[Some Mathematical Ideas in Bayesian Statistics#2. 边际化（Marginalization）和先验信息的利用|Example 2]], where $\theta=\sum_iY_i/N$ is proportion of happy people in entire population, i.e. out belief about $\theta$.
## Sampling Model

$Y_1,\cdots,Y_n$ are conditionally independent given $\theta$ 
$$
p(y_1,\cdots,y_n|\theta)=\prod_{i=1}^np(y_i|\theta)=(1-\theta)^{N-\sum y_i}\theta^{\sum y_i}
$$

>[!Question] 
>Are $Y_1,\cdots,Y_n$ are independent?
>
>**Answer:**
>
>They are not independent.
>$$
>(\underset{\text{observed}}{\underline{Y_1,\cdots,Y_n}},\underset{\text{hidden}}{\underline{\theta}})
>$$
But $Y_1,\cdots,Y_n|\theta$ are independent.

If our sampling model is $p(\sum y_i|\theta)$ , then we can consider Binomial Distribution as our sampling model.

### Prior Distribution

For $\theta$ is some unknown number between $0$ and $1$, so we can use uniform distribution as our prior (equal weight to all values of $\theta$).
$$
\theta\sim U[0,1]=Beta(1,1)
$$
According to the posterior distribution under below, we can consider conjugate prior, which is
#### Conjugate

>[!Note] Definiton (Conjugate)
>
>A class of $\mathcal{P}$ of prior distributions for $\theta$ is called **conjugate** for a sampling model $p(y|\theta)$ if
>$$
>p(\theta)\in\mathcal{P}\Rightarrow p(\theta|y)\in \mathcal{P}
>$$

So we can consider [[Probability Reference#Beta Distribution|Beta distribution]] $Beta(a,b)$ as our conjugate prior. Using conjugate prior will simplify our calculation and analysis.

### Posterior Distribution

Using [[Probability Reference#Bayes' Rule/ Bayes' Theorem|Bayes' Rule]], [[Some Mathematical Ideas in Bayesian Statistics#2. 边际化（Marginalization）和先验信息的利用|Marginalization]] and [[Some Mathematical Ideas in Bayesian Statistics#3. 正则化（Normalization）|Normalization]], we can get
$$
p(\theta|y)=\frac{p(y|\theta)p(\theta)}{p(y)}\propto \theta^{a+y-1}(1-\theta)^{b+n-y-1}
$$
Then we have
$$
p(\theta|y)=\frac{\Gamma(a+b+n)}{\Gamma(a+y)\Gamma(b+n-y)}\theta^{a+y-1}(1-\theta)^{b+n-y-1}
$$
which means $\theta|y\sim Beta(a+y,b+n-y)$ .
We can calculate posterior expectation as below
$$
\begin{array}{}
E(\theta|Y=y)&=&\frac{a+y}{(a+y)+(b+n-y)}\\
&=&\frac{a+y}{a+b+n}\frac{a}{a+b}+\frac{n}{a+b+n}\frac{y}{n}\\
&=& w_1\times \text{Prior Mean}+w_2\times\text{Data Mean}
\end{array}
$$
- If $n>>a,b$, then $E(\theta|Y=y)=\frac{y}{n}=\text{Data Mean}$
- If $a+b>>n$, then $E(\theta|Y=y)=\frac{a}{a+b}=\text{Prior Mean}$

### Interpreting Parameters

|          $a$          |          $b$          |       $a+b$       |
| :-------------------: | :-------------------: | :---------------: |
| prior number of $1$'s | prior number of $0$'s | prior sample size |

|           $y$            |          $n-y$           |         $n$          |
| :----------------------: | :----------------------: | :------------------: |
| observed number of $1$'s | observed number of $0$'s | observed sample size |

Actually under conjugate prior, we can write posterior distribution as
$$
\text{posterior}=\begin{array}{}
Beta(\text{prior number of $1$'s}+\text{observed number of $1$'s},\\\text{prior number of $0$'s}+\text{observed number of $0$'s})
\end{array}
$$

### Predictive Distribution

By [[Probability Reference#Conditional Independence|Conditional Independence]] and posterior distribution, we can get
$$
\begin{array}{rl}
\Pr(\tilde{Y}=1|y_1,\cdots,y_n)&=\int \Pr(\tilde{Y}=1,\theta|y_1,\cdots,y_n)d\theta\\
&=\int\Pr(\tilde{Y}=1|\theta,y_1,\cdots,y_n)p(\theta|y_1,\cdots,y_n)d\theta\\
&=\int \Pr(\tilde{Y}=1|\theta)p(\theta|y_1,\cdots,y_n)d\theta\\
&=\int\theta p(\theta|y_1,\cdots,y_n)d\theta\\
&=E[\theta|y_1,\cdots,y_n]\\
&=\frac{a+\sum y_i}{a+b+n}
\end{array}
$$
Therefore, $\tilde{Y}|y_1,\cdots,y_n\sim Bernoulli(\frac{a+\sum y_i}{a+b+n})$ depends on our observed data.

**Notice:** $\tilde Y$ is not independent with $Y_1,\cdots,Y_n,\theta$ , for $Y_1,\cdots,Y_n\rightarrow\theta\rightarrow\tilde Y$ .

### Other Properties

- $y_1,\cdots,y_n$ joint distribution or $\sum y_i$ distribution is not Beta or Bernoulli or Gamma or Binomial or Negative Binomial;
- $y_i$'s distribution is Bernoulli, which is $y_i\thicksim Bernoulli(\frac{a}{a+b})$ 
- We can prove that $\sum_{i=1}^ny_i$ is a sufficient statistic for $\theta$ and $p(y_1,\cdots,y_n|\theta)$. We also have $p(\theta|y_1,\cdots,y_n)=p(\theta|\sum_{i=1}^ny_i)$, which means that the information contained in $\{Y_1=y_1,\cdots,Y_n=y_n\}$ is the same as the information contained in $\{Y=\sum Y_i=\sum y_i=y\}$.
---
# Poisson Model

Some measurements, such as a person's number of children or number of friends, have values that are whole numbers. In these cases our sample space is $\mathcal{Y}=\{0,1,2,\cdots\}$.

Poisson family of distributions has a "mean-variance relationship": larger mean than another, larger variance as well.

## Sampling Model

$$
Y_i|\theta\overset{iid}\sim Poisson(\theta)\Rightarrow p(y_i|\theta)=\frac{\theta^{y_i}}{y_i!}e^{-\theta}
$$
$$
\sum_{i=1}^nY_i\text{ is a sufficient statistic}\thicksim Poisson(n\theta)
$$

## Conjugate Prior

$$
\theta\thicksim Gamma(a,b)
$$
see [[Probability Reference#Gamma Distribution|Gamma Distribution]].

## Posterior Distribution

$$
\begin{array}{}
p(\theta|y_1,\cdots,y_n)&\propto& p(\theta)p(y_1,\cdots,y_n|\theta)\\
&\propto& \theta^{a-1+\sum y_i}e^{-(b+n)\theta}
\end{array}
$$
Therefore, we have
$$
\theta|y_1,\cdots,y_n\thicksim Gamma(a+\sum y_i,b+n)
$$
- $E[\theta|data]=\frac{b}{b+n}\frac{a}{b}+\frac{n}{b+n}\frac{\sum y_i}{n}$ 
	- when $n\rightarrow\infty$, $E\rightarrow \bar Y=\text{Data Mean}$
	- when $b\rightarrow \infty$, $E\rightarrow \frac{a}{b}=\text{Prior Mean}$

## Interpreting Parameters

|                       a                       |                               b                                |
| :-------------------------------------------: | :------------------------------------------------------------: |
| the sum of counts from $b$ prior observations | the number of prior observations(strength of our prior belief) |

|                    $\sum y_i$                    |                  n                  |
| :----------------------------------------------: | :---------------------------------: |
| the sum of counts from $n$ sampling observations | the number of sampling observations |

## Predictive Distribution

We can proved that
$$
\tilde{y}|y_1,\cdots,y_n\thicksim \text{Negative Binomial}
$$
see also in [[Probability Reference#Negative Binomial Distribution|Negative Binomial]].
