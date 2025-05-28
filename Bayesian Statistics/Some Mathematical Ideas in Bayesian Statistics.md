
Bayes' rule does not determine what our beliefs should be after seeing the data, it only tells us how they should change after seeing the data. (实际上就是我们会根据样本数据对我们先验的belief进行修正，更关注修正后的后验)

---
# 1. 对已有结果进行处理转化为我们想要的已知信息

根据贝叶斯统计中的一些思想，我们会关注后验分布（Posterior Distribution）的一些具体的统计性质，如分布类型，相关的数学特征等。而在贝叶斯统计的研究与思想中，我们并不会将后验分布与先验分布（Prior Distribution）、样本分布（Sampling Model）割裂开单独分析，而会更加注意三者分布之间的联系，如共轭先验（Conjugate Prior）以及数学特征上的一些联系。所以我们需要做到的就是如何将后验分布得到的一些结果进行适当的转化从而将其与先验分布、样本分布联系起来。当前总结的一些处理思路如下：

- **注意寻找结果中与另外两个分布具有联系的参数，这些参数会引导我们将结果与特定分布相联系；**
- **注意将相关联参数向另外两个分布的一些数字特征进行转化**，如出现了样本值，我们就可以考虑向样本均值转化；出现了先验分布的参数，那么就考虑如何将这些参数向分布的数字特征转换；

>[!example] Example 1
>在计算一座城市的感染人群 $Y=\{0,1,\cdots,n\}$ 与感染率 $\theta\in[0,1]$ 之间的关系时，我们已知 $\theta\thicksim Beta(a,b)$，$Y|\theta \thicksim Binomial(n,\theta)$，那么根据条件概率公式，我们可以得到后验分布 $p(\theta|Y)\thicksim Beta(a+y,b+n-y)$ ，其中 $n$ 为样本容量，$y$ 为观测值，那么我们可以得到后验分布的均值为：
>$$
>\begin{array}{lcl}
>E[\theta|y] &=& \frac{a+y}{a+y+b+n-y}\\
>&=& \frac{a}{a+b+n}+\frac{y}{a+b+n}\\
>&=& \frac{a+b}{a+b+n}\frac{a}{a+b} + \frac{n}{a+b+n}\frac{y}{n}\\
>&=& \frac{a+b}{a+b+n}\cdot\text{Prior Mean} + \frac{n}{a+b+n}\cdot\text{Data Mean}
>\end{array}
>$$
>这样，我们就将后验分布的期望与先验分布和样本模型的均值联系起来，那么我们考虑 $n\rightarrow \infty,\quad E[\theta|y]\rightarrow \theta_0\,(\text{Prior mean})$，$a+b \rightarrow \infty,\quad E[\theta|y]\rightarrow \bar{y}\,(\text{Data mean})$ 。$\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\square$

所以在学习与分析的过程中，我们更需要**注重先验分布、样本模型、后验分布三者之间的联系**，**切勿割裂**开单独分析！

---
# 2. 边际化（Marginalization）和先验信息的利用

>[!question] Why?
>我们为什么可以在贝叶斯推断的过程中省略一些参数来简化分析？
>
>**Answer：**
>
>在推断过程中可以省略一些参数的原因在于，对于一些是固定为常数的参数，或者是不包含任何分布信息的参数，我们并不关心这些参数，而是更关心于一些包含分布信息的参数，可以用来直接确定分布类型的主体，或者是充分统计量（Sufficient Statistics）等。所以省略这些无用或者不包含分布信息的参数，可以大大简化我们的分析过程，方便我们直接定性后验分布究竟为什么分布。

- 贝叶斯推断中，如果模型包含多个参数，但我们只对其中一部分参数感兴趣，那么可以通过**边际化（即对不感兴趣的参数进行积分）** 来消除这些参数的影响，集中于感兴趣的参数。
- 实际上对不感兴趣的参数进行积分是一种分析方式，在一些分析与推断的过程中，我们会更倾向于**考虑概率密度函数（PDF）之间定性的正比关系**，即带入先验分布与样本分布，省略到固定的以及不包含分布信息的参数，从而利用分布 PDF 之间的正比关系，确定后验分布的具体分布类型与形式。
- 明确pdf中哪个变量是核心，要保证这个核心变量在前后推断中一一对应同时存在。

为了更直观地理解边缘化的技巧，我们举例如下：
>[!example] Example 2
>
>在二项模型（Binomial Model）中，我们分析样本中感到快乐的人的分布，定义随机变量
>
>$$
>Y_i=\begin{cases}
>1,\quad \text{if i-th person is happy}\\
>0,\quad \text{if i-th person is unhappy}
>\end{cases}
>$$
>
>我们可以得到 $Y_i|\theta\overset{iid}{\thicksim} Bernoulli(\theta)$，先验分布 $\theta\thicksim Uniform(0,1)=Beta(1,1)$ (equal weight to all values of $\theta$)，那么得到后验分布与样本分布、先验分布之间的正比关系为
>
>$$
>\begin{array}{lcl}
>p(\theta|y_1,\cdots,y_n) &=& \frac{p(y_1,\cdots,y_{129}|\theta)\cdot p(\theta)}{p(y_1,\cdots,y_{129})}\\
>&=& p(y_1,\cdots,y_{129}|\theta)\cdot \frac{1}{p(y_1,\cdots,y_{129})}\quad(\text{边际化后一项})\\
>&\propto& p(y_1,\cdots,y_{129}|\theta)\\
>&=& \theta^{\sum y_i}(1-\theta)^{129-\sum y_i}
>\end{array}
>$$
>注意到 $p(y_1,\cdots,y_{129})$ 是一个与分布主体 $\theta$ 无关的一个常数，因此我们并不关心它，所以对它进行边际化处理，只需考虑主体之间的正比关系即可。$\square$

---
# 3. 正则化（Normalization）

在贝叶斯推断中，我们往往将标准化与上文中的[[Some Mathematical Ideas in Bayesian Statistics#2. 边际化（Marginalization）和先验信息的利用|边际化(Marginalization)]]共同使用：
- 通过边际化得到 Posterior Distribution 与分布主体的正比关系
- 观察分布主体的数字特征，将其与已知分布联系，确定分布的参数
- 利用 $\int \,\text{PDF}\, d\theta \equiv 1$ ，计算得到后验分布中的未知系数，从而得到完整的概率密度函数

这种标准化的思想不止可以应用在贝叶斯推断中，只要满足一下的场景，我们都可以尝试使用正则化的技巧：
- 已知结果恒等于常数的公式
- 目标公式与上公式具有相同的主体部分
- 利用已知公式，恒等变换，计算目标公式中的未知项

为了更直观地理解边缘化的技巧，我们同样以 Example 2为例：

>[!example] Example 2 Continued
>针对 Example 2的例子，代入观测值 $\sum_{i=1}^{129}y_i=118$，我们可以得到
>
>$$
>p(\theta|y_1,\cdots,y_n) \propto \theta^{118}(1-\theta)^{11}
>$$
>
>注意到上述正比关系右侧为 $Beta(119,12)$ 的主体部分，利用正则化的思想，我们可以得到
>$$
>\int_0^1 \theta^{118}(1-\theta)^{11} d\theta=\frac{\Gamma(119)\Gamma(12)}{\Gamma(131)}
>$$
>
>因此，我们得到 $p(\theta|y_1,\cdots,y_{129}=\frac{\Gamma(131)}{\Gamma(119)\Gamma(12)}\theta^{118}(1-\theta)^{11}=Beta(119,12)$ $\square$

---
# 4. 充分统计量（Sufficient Statistics）

在贝叶斯统计中，充分统计量是一个我们绕不开的话题。其定义如下:

>[!note] Definition (Sufficient Statistics)
>
>A Statistics $T(X_1,\cdots,X_n)$ is said to be **sufficient** for $\theta$ if the conditional distribution of $X_1,\cdots,X_n$ , given $T=t$ , does not depend on $\theta$ for any value of $t$.

充分统计量的意义在于，倘若给定了其确定的值，那么我们就无法从 $X_1,\cdots,X_n$ 中获得关于 $\theta$ 的任何信息，也就是说充分统计量  $T(X_1,\cdots,X_n)$ 包含了关于参数 $\theta$ 的所有信息。

但这并不意味着我们可以直接忽略掉 $X_1,\cdots,X_n$ ，只保留 $T(X_1,\cdots,X_n)$ ：因为在模型检验的过程中，这些分量可能会表明模型并不合适，作为离群点给我们一些关于模型的启发。

那么我们应该如何找到一个分布的充分统计量呢？下面的因子分解定理（Fisher-Neyman Factorization Theorem）为我们提供了一个寻找充分统计量的强力工具。

>[!note] Theorem (Fisher-Neyman Factorization Theorem)
>
>A necessary and sufficient condition for $T(X_1,\cdots,X_n)$ to be sufficient for a parameter $\theta$ is that the joint probability function (density function or frequency function) factors in the form
>$$
>f(x_1,\cdots,x_n|\theta)=g[T(x_1,\cdots,x_n),\theta]h(x_1,\cdots,x_n)
>$$

**Proof:**
- Discrete:
	Let $X=(X_1,\cdots,X_n),\,x=(x_1,\cdots,x_n)$, then we have
	$$
	P(T=t)=\sum_{T(x)=t}P(X=x)=g(t,\theta)\sum_{T(x)=t}h(X)
	$$
	- Sufficiency:
		$$
		\begin{array}{lcl}
		P(X=x|T=t)&=&\frac{P(X=x,T=t)}{P(T=t)}\\
		&=& \frac{g(t,\theta)h(X)}{g(t,\theta)\sum_{T(x)=t}h(X)}\\
		&=& \frac{h(x)}{\sum_{T(x)=t}h(x)}
\end{array}
		$$
		This conditional distribution does not depend on $\theta$. Sufficiency has been proved.		
	- Necessity:
		Suppose that the conditional distribution of $X$ given $T$ is independent of $\theta$. Let
		$$
		\begin{array}{lcl}
		P(X=x|\theta)&=&P(X=x,T=t|\theta)\\
		&=& P(T=t|\theta)\cdot P(X=x|T=t,\theta)\\
		&=& P(T=t|\theta)\cdot P(X=x|T=t)
	\end{array}
		$$
		Let $g(t,\theta)=P(T=t|\theta),\,P(X=x|T=t)=h(x)$, then we can get
		$$
		P(X=x|\theta)=g(t,\theta)\cdot h(x)
		$$
		Necessity has been proved. 

- Continuous:


---
# 5. 条件期望的塔式性质/迭代法则，全方差定理(Law of Total Variance)与ANOVA

条件期望的塔式性质其实也是全期望定理(Law of Total Expectation)针对多个条件下的推广。
$$
E[E[U|V,W]|W]=E[U|W]
$$
直观上，我们可以将 $E[U|V,W]$ 视为关于随机变量 $V,W$ 的随机变量，即 $E[U|V,W]=g(V,W)$。那么 $E[E[U|V,W]|W]=E[g(V,W)|W]$ 相当于给定 $W$，计算条件分布 $g(V,W)|W$ 的期望，因此我们需要对 $V$ 进行积分，从而消除了 $V$ 的影响，即 $E[E[U|V,W]|W]=E[g(V,W)|W]=h(W)$ 是关于 $W$ 的随机变量。

全方差定理作为类似全期望定理(Law of Total Expectation)的类似推广，同样给出了一种区别于定义的计算方差的方法。这种方法同时也具有全期望定理灵活的应用方式，即可以任意选取对于我们计算有利的事件作为条件，这为我们的计算能够带来极大的便利。

事实上，全方差定理与ANOVA都是对与方差的分解，而且具有相同的分解思想，即：将总方差分解为组间方差和组内方差。
$$
Var(U)=E(Var[U|V])+Var(E[U|V])
$$
其中，我们对分解有如下解释：

 - $Var(U)$, 即 SST，**总离差平方和**，表示**所有观测值与总体均值**之间的离差平方和,反映了总体数据的离散程度；
- $E(Var[U|V])$, 即 SSW，**组内离差平方和**，表示**组均值与总体均值**之间的离差平方和，反映了组间的差异；
- $Var(E[U|V])$, 即 SSB，**组间离差平方和**，表示**各组内数据点与该组均值**之间的离差平方和，反映了组内的差异。

全方差定理的思想与ANOVA的思想相同，但ANOVA会更注重基于方差分解的假设检验来进行统计推断。

---
# 6. Pro-Move: 对比较复杂的参数进行换元简化计算

在贝叶斯统计中计算后验分布的具体形式时，我们往往会遇到一些比较复杂的分布形式，这个时候冗杂的参数形式会为我们的计算带来非常多的阻碍。一种在贝叶斯统计中的 Pro-Move 就展现了它的优势性，即将一些特殊的参数进行换元从而简化计算。如在考虑后验分布的主体部分正比关系时，出现了完全平方公式的形式，这个时候我们就可以主动将一次项系数，二次项系数进行换元，利用换元后的参数构建完全平方式，并在构建的过程中[[Some Mathematical Ideas in Bayesian Statistics#2. 边际化（Marginalization）和先验信息的利用|边际化(Marginalization)]]其他非重要参数，从而简化计算。下面我们以推导已知方差 $\sigma^2$ 的正态模型后验分布过程中对条件先验分布的参数换元为例，说明这个技巧：

>[!Example] Example 3: Conjugate Prior (Conditional) Distribution of Normal Model
>
>We have known that
>$$
>\theta|\sigma^2\thicksim N(\mu_0,\tau_0^2)
>$$
>Then we have
>$$
>\begin{array}{lcl}
>p(\theta|\sigma^2)&\propto& \exp[-\frac{1}{2\tau_0^2}(\theta-\mu_0)^2]\\
>&=&\exp[-\frac{1}{2}(\frac{\theta^2}{\tau_0^2}-\frac{2\theta\mu_0}{\tau_0^2}+\frac{\mu_0^2}{\tau_0^2})]\xlongequal{a=\frac{1}{\tau_0^2},b=\frac{\mu_0}{\tau_0^2}} \exp[-\frac{1}{2}(a\theta^2-b\theta+c)]\\
>&\propto& \exp[-\frac{1}{2}(\frac{\theta-b}{1/\sqrt{a}})^2]\quad\text{(Marginalization)}
>\end{array}
>$$

这样在经过换元后，计算后验分布时我们就可以使用 $a,b$ 来代替长参数，从而简化计算。

---
## 7. odds 的引入

在实际分析中，我们往往会遇到比较两个概率的大小，从而对参数进行筛选。这时候如果使用常规的做差方法并不具有可行性，因为我们有时候并不能得到一个确切的后验分布的形式。因此这时候我们引入比率(odds)的概念，通过贝叶斯定理将后验概率分解为先验概率与样本概率的乘积，从而省去了计算后验分布的过程。
$$

\frac{\Pr(H_i|E)}{\Pr(H_j|E)}=\frac{\Pr(E|H_i)}{\Pr(E|H_j)}\times\frac{\Pr(H_i)}{\Pr(H_j)}=\text{``Bayes Factor"}\times\text{``prior beliefs"}

$$

Using ratio will also help us get sufficient statistic easily.

---
# 8. Thin out (Thinning) 在马尔可夫链中的应用

Thin out 是指在 MCMC 采样过程中，每隔一定间隔（例如每隔 $k$ 个样本）保留一个样本，而丢弃中间的样本。这个处理的**目的**是，通过减少样本之间的自相关性，使得保留的样本更加独立，从而提高样本的质量和统计推断的准确性。

在mcmc生成的样本过程中，生成的样本通常是自相关的。这意味着我们在模拟的过程中，往往会生成很多无用的数据。而这些无用数据会导致样本的有效性降低，计算效率下降。通过 thin out，可以减少样本之间的自相关性，使得保留的样本更接近独立同分布（i.i.d.），从而提高样本的有效性和计算效率。

假设我们有一个 MCMC 采样过程，生成了一个长度为 $N$ 的样本序列 $\{x_1,x_2,…,x_N\}$。thin out 的实现步骤如下：

1. **选择间隔 $k$**：确定每隔多少个样本保留一个样本。例如，$k=10$ 表示每隔 10 个样本保留一个样本。

2. **保留样本**：从原始样本序列中每隔 kk 个样本保留一个样本。例如，保留的样本为 $\{x_1,x_{k}+1,x_{2k}+1,… \}$。

3. **丢弃中间样本**：丢弃未被保留的样本。

最终，保留的样本数量为 $N/k$（假设 $N$ 是 $k$ 的倍数）。

## 优点：

- **减少自相关性**：通过丢弃中间样本，保留的样本更加独立，提高了样本的有效性。
- **节省存储空间**：减少了需要存储的样本数量，降低了存储和计算成本。
- **简化后续分析**：由于样本之间的自相关性降低，后续的统计分析（如均值估计、方差估计等）更加可靠。
## 缺点：

- **信息损失**：丢弃的样本中可能包含有用的信息，尤其是 $k$ 过大时。
- **效率降低**：如果 $k$ 过大，可能导致保留的样本数量过少，从而降低统计推断的精度。

---
