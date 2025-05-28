#Probability 
# Definition/ Theorems/ Formula

## Independence and Uncorrelatedness

$$
\begin{array}{}
\text{Independence} \overset{√}{\underset{\text{x}}{\rightleftharpoons}} \text{Uncorrelatedness}\\
\text{Independence of normal RV}\rightleftharpoons \text{Uncorrelatedness}
\end{array}
$$

>[!example]
>Consider $X\thicksim N(0,1),\,Y=X^2+1$ , it is obvious that $X$ and $Y$ are not independent, but we can calculate that
>$$
>cov(X,Y)=E(X^3)=0
>$$
>which means that $X$ and $Y$ are uncorrelated, but not independent.$\square$

## Partition

Let $\mathcal{H}=\text{a set of all possible truths(or }\Omega)$, then $\{H_i\}^k_{i=1}$ forms a **partition** of $\mathcal{H}$ if $H_i\cap H_j=\Phi,i\neq j$ and $\cup_{i=1}^k H_i=\mathcal{H}$.

### Rule of Total Probability
$$
\sum_{k=1}^K\Pr(H_k)=1
$$
where $\{H_1,\cdots,H_k\}$ is a partition of $\mathcal{H}$, $\Pr(\mathcal{H})=1$.

## Rule of Marginal Probability
$$
\Pr(A)=\sum_{i=1}^K\Pr(A\cap H_k)=\sum_{i=1}^K\Pr(A|H_k)\Pr(H_k)
$$

## Bayes' Rule/ Bayes' Theorem
$$
\Pr(H_i|A)=\frac{\Pr(A\cap H_i)}{\Pr(A)}=\frac{\Pr(A|H_i)\Pr(H_i)}{\sum_j\Pr(A|H_j)\Pr(H_j)}
$$

## Conditional Independence

Two events $A,B$ are conditionally independent given event $C$ if
$$
\Pr(A\cap B|C)=\Pr(A|C)\Pr(B|C)
$$
We can also have
$$
\frac{\Pr(A\cap B|C)}{\Pr(B|C)}=\Pr(A|C)\Rightarrow\Pr(A|B\cap C)=\Pr(A|C)
$$
which means that $B$ has **no information** on $A$.

## Law of Total Expectation

For random variables $X,Y$
$$
E[X]=E[E[X|Y]]
$$
More general, for $3$ random variables $X,Y,Z$
$$
E[E[X|Y,Z]|Y]=E[X|Y]
$$
	直观理解：
	$E[X|Y,Z]=g(Y,Z)$ 是关于 $Y,Z$ 的随机变量，$E[g(Y,Z)|Y]$ 相当于给定 $W$，求条件分布 $g(Y,Z)|Y$ 的期望，故对 $Z$ 积分，消除了 $Z$ 的影响。$E[g(Y,Z)|Y]=h(Y)$ 是关于 $Y$ 的随机变量。

## Law of Total Variance

For $3$ random variables $U,V,W$,
$$
Var(U)=E(Var[U|V])+Var(E[U|V])
$$
see [[Some Mathematical Ideas in Bayesian Statistics#5. 条件期望的塔式性质/迭代法则，全方差定理(Law of Total Variance)与ANOVA|Law of Total Variance and ANOVA]].

## Inequality
### Chebyshev Inequality

For nay random variable $X$ with mean $\mu$ and variance $\sigma^2$, we have
$$
\mathbb{P}(|X-\mu|\geq x)\leq\frac{\sigma^2}{x^2}
$$

### Markov Inequality

Let $D^2=(X-\mu)^2$, then we have
$$
\mathbb{P}(D^2\geq x^2)\leq \frac{\sigma^2}{x^2}
$$
---
# Distributions

## Continuous Distributions

![[Continuous distributions 1.png]]
![[Continuous Distributions 2.png]]

### Beta Distribution

For random variable $X\thicksim Beta(a,b)$ , then we have
$$
\text{PDF: }f(x)=\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}x^{a-1}(1-x)^{b-1}
$$
where $x\in [0,1]$ or $x\in (0,1)$ . And we can also get the numeric characteristics
$$
E[X]=\frac{a}{a+b}\quad Var[X]=\frac{ab}{(a+b)^2(a+b+1)}\quad Mode[X]=\frac{a-1}{a+b-2}
$$
- Uniform distribution $U[0,1]$ is also Beta distribution $Beta(1,1)$.

### Gamma Distribution

- Gamma Function
	$\Gamma(z)=\int_0^{\infty}t^{z-1}e^{-t}dt$
	- $\Gamma(z+1)=z\Gamma(z)$
	- $\Gamma(n)=(n-1)!$
	- $\Gamma(1)=(1-1)!=1$
- For random variable $X\thicksim Gamma(a,b)$, we have
$$
\text{PDF: }f(x)=\frac{b^a}{\Gamma(a)}x^{a-1}e^{-bx}
$$
	where $x\in(0,\infty)$. And we can also get the numeric characteristics
$$
E[X]=\frac{a}{b}\quad Var[X]=\frac{a}{b^2}\quad Mode[X]=\begin{cases}
(a-1)/b,\quad\text{if }a>1\\\quad\quad0\quad\,\,,\quad \text{if }a\leq1 
\end{cases}
$$
	- $X\thicksim Gamma(a,b)\Rightarrow cX\thicksim Gamma(a,\frac{b}{c})$

### Three Distributions: $\chi^2$, Student-$t$, $F$
#### Chi-squared $\chi^2$

If $z_1,\cdots,z_n\overset{iid}{\thicksim}N(0,1)$, then
$$
\sum_{i=1}^nz_i^2\thicksim \chi^2_n
$$
where $n$ is the degree of freedom.

#### $F$ distribution

If $U\thicksim \chi^2_k,\,V\thicksim\chi^2_m,\, U\perp\!\!\!\perp V$, then
$$
	\frac{U/k}{V/m}\thicksim F_k,m
	$$
where $k,m$ are degrees of freedom.

#### Student-$t$ distribution

If $Z\thicksim N(0,1),\,U\thicksim \chi^2_k,\,Z\perp\!\!\!\perp U$, then
$$
\frac{Z}{\sqrt{U/k}}\thicksim t_k
$$
where $k$ is degree of freedom.

#### Properties

- If $Y_1,\cdots,Y_n\overset{iid}{\thicksim} N(\mu,\sigma^2),\,s^2=\frac{1}{n}\sum_{i=1}^n(Y_i-\bar Y)^2$, $\bar Y \perp\!\!\!\perp s^2$ (we can prove this), then
$$
\begin{array}{}
\frac{\bar Y-\mu}{\sqrt{s^2/n}}\thicksim t_{n-1}\\
\frac{(n-1)s^2}{\sigma^2}\thicksim \chi^2_{n-1}
\end{array}
$$
- When student-$t$ distribution and $F$ distribution have **same degree of freedom**(for **$F$ distribution's first degree of freedom**), then for the statistics of two distribution, we have
	$$
	t^2_n=F_{1,n}
	$$
	We have proved this in linear regression part.

### Logistic Distribution

PDF of standard logistic random variable
$$
f_L(\varepsilon_L)=\frac{\exp(\varepsilon_L)}{[1+\exp(\varepsilon_L)]^2}
$$
CDF of standard logistic random variable
$$
F_L(\varepsilon_L)=\int_{-\infty}^{\varepsilon_L}f_L(\mu)d\mu=\frac{\exp(\varepsilon_L)}{1+\exp(\varepsilon_L)}
$$
Properties of $\varepsilon_L$:
- $E(\varepsilon_L)=0$
- $Var(\varepsilon_L)=\frac{\pi}{\sqrt{3}}$

### Gumbel Distribution

PDF of Gumbel random variable
$$
f(x;\mu,\beta)=\frac{1}{\beta}\exp^{-\frac{x-\mu}{\beta}}\exp^{-\exp^{-\frac{x-\mu}{\beta}}}
$$
CDF of Gumbel random variable
$$
F(x;\mu,\beta)=\exp^{-\exp^{-\frac{x-\mu}{\beta}}}
$$

## Discrete Distributions
### Negative Binomial Distribution

For random variable $X\thicksim NB(r,p)$, we have
$$
\text{PMF: }\Pr(X=k)=\begin{pmatrix}k+r-1\\k\end{pmatrix}(1-p)^kp^{r}
$$
where $\{X=k\}=$The $r_{th}$ successful occurrence occurred in the $k_{th}$ experiment. 

And we can also get the numeric characteristics
$$
E[X]=\frac{r(1-p)}{p}\quad Var[X]=\frac{r(1-p)}{p^2}
$$

### Poisson Distribution

PDF of Poisson Distribution $Y\thicksim Poisson(\mu)$
$$
f(y)=\frac{\mu^y\exp(-\mu)}{y!},y=0,1,2,\cdots
$$

- $E(Y)=\mu$
- $Var(Y)=\mu$ 

## Numerical Characteristics

### Expectation, Covariance and Variance

- Expectation is a measure of centrality, "weight average" of RV;
- Covariance measures how the two variables vary "jointly";
- Variance is a measure of spread of RV.

![[variance distributions.png]]

---
# Transformation for Random Variable

## Change of Variable Formula

For known random variable $Y$ with pdf $p(y)$, consider random variable $\eta=g(y)$, we can calculate
$$
p(\eta)=p(y)|\frac{dy}{d\eta}|=p(g^{-1}(\eta))|\frac{dy}{d\eta}|
$$

## General Transformation for Random Variables

Similar to Change of Variable Formula, we can get general transformations in high dimension.

Consider random vector $\mathbf{X}$ with p.d.f. $f_\mathbf{X}(\mathbf{x})$, we have general transformations $\mathbf{x}\mapsto g(\mathbf{x})$ 
$$
\begin{pmatrix}
x_1\\x_2\\\vdots\\x_n
\end{pmatrix}\mapsto \begin{pmatrix}
g_1(\mathbf{x})\\g_2(\mathbf{x})\\\vdots\\g_n(\mathbf{x})
\end{pmatrix}
$$
Let $\mathbf{Z}=g(\mathbf{X})$, we have
$$
f_\mathbf{Z}(\mathbf{z})=f_\mathbf{X}(g^{-1}(\mathbf{z}))\vert J_\mathbf{z}(g^{-1})\vert
$$
where $\vert J_\mathbf{z}(g^{-1})\vert=1/\vert J_\mathbf{x}(g)\vert$.

---
# Sampling Theorem of Normal Distribution

>[!note] Theorem
>If $Y_1,\cdots,Y_n\overset{iid}{\thicksim}N(\mu,\sigma^2)$ , then we have
>- If $\sigma^2$ is known, $\frac{\bar Y-\mu}{\sigma/\sqrt{n}}\thicksim N(0,1)$
>- If $\sigma^2$ is unknown, $\frac{\bar Y-\mu}{s/\sqrt n}\thicksim t_{n-1}$
>- If $\sigma^2$ is known, $\frac{(n-1)s^2}{\sigma^2}\thicksim \chi^2_{n-1}$
>- $s^2\perp\!\!\!\perp \bar Y$

Proof: Construct a transformation matrix about $Y_1,\cdots,Y_n$, starting with the orthogonal transformation of line a $(\frac{1}{\sqrt n},\cdots,\frac{1}{\sqrt n})_{1\times n}$ .

---
# Hypothesis Testing and Confidence Interval

## Intuition

Hypothesis test: $\mu$ (null value) is given;
Confidence Interval: $\mu$ is not given.

## p-value

We consider the value of **the test statistics with samples**.

- **p-value**: reflected in the image as the **observation value of statistics**($t^*$ or observed value);
- **rejection region**: reflected in the image as the **quantile of the test distribution**(cv-critical value or $u_\alpha$)

Note that **whether the test is two-sided test or one-sided test**. (The following is only an example, and specific situations need to be analyzed on a case by case basis)
$$
\begin{array}{}
\text{Two-sided}:\text{p-val}=\Pr(|T|>|t^*|)=2\Pr(T>|t^*|)\\
\text{One-sided}:\text{p-val}=\Pr(T>t^*)
\end{array}
$$


