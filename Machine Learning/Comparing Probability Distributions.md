
These $3$ definitions are used to evaluate the difference between two distributions.
# Total Variation Distance

>[! note] Definition (Total Variation Distance)
>Let $P$ and $Q$ be two probability distributions on $\mathcal{R}$. Their total variation distance is defined by
>$$
>\operatorname{D}_{\text{var}}(P,Q)=\sup_A(P(A)-Q(A))
>$$
>where the supremum is taken over all measurable sets $A$.
- $\operatorname{D}_{\text{var}}(P,Q)=\max_A|P(A)-Q(A)|$
- Total variation distance is also a kind of distance, which satisfies all properties of distance (Non-negativity, Symmetry, Triangle Inequality).
- 变分距离实际上就是寻找一个使得两个分布之间差异最大的set。

>[! note] Lemma
There exists a measurable set $A_0$ such that
>$$
>\operatorname{D}_{\text{var}}(P,Q)=P(A_0)-Q(A_0)
>$$
>and for all $B$,
>$$
>P(B\cap A_0)\geq Q(B\cap A_0) \Leftrightarrow P(B\cap A^c_0)\leq Q(B\cap A^c_0)
>$$
>- $\mathcal{R}$ is finite: $A_0=\{x\in \mathcal{R}:P(x)\geq Q(x)\}$；
>- p.d.f. exists: $A_0=\{x\in \mathcal{R}:\varphi_P(x)\geq \varphi_Q(x)\}$ where $P,Q<< \mu$.
>- 证明的核心在于利用$A\cap A_0,A^c\cap A_0$进行不等式的放缩，从 $P(A)-Q(A)$ 向 $P(A_0)-Q(A_0)$ 转化

The proposition below gives some other notations of $\operatorname{D}_{\text{var}}(P,Q)$.

>[! note] Proposition
>1. If $P,Q$ have a density $\varphi_P,\varphi_Q$ with respect to some positive measure $\mu$, then$$\begin{equation}
\begin{aligned}
\text{Discrete:}\,\operatorname{D}_{\text{var}}(P,Q)&=\frac{1}{2}\sum_{x\in\mathcal{R}}|P(x)-Q(x)|\\
\text{Continuous:}\,\operatorname{D}_{\text{var}}(P,Q)&=\frac{1}{2}\int_{\mathcal{R}}|\varphi_P(x)-\varphi_Q(x)|\mu(dx)
\end{aligned}
\end{equation}$$                                                                                         
>2. For general $\mathcal{R}$, $$
\operatorname{D}_{\text{var}}(P,Q)=\sup_f\left(\int_\mathcal{R}f(x)P(dx)-\int_\mathcal{R}f(x)Q(dx)\right)$$where $f:\mathcal{R}\rightarrow[0,1]$ is measurable.                        
>3. If $f:\mathcal{R}\rightarrow \mathbb{R}$ is bounded, define the maximal oscillation of $f$ by$$\operatorname{osc}(f)=\sup\{f(x)-f(y):x,y\in\mathcal{R}\}$$. Then$$\operatorname{D}_{\text{var}}(P,Q)=\sup_f\left\{\int_\mathcal{R}f(x)P(dx)-\int_\mathcal{R}f(x)Q(dx):\operatorname{osc}(f)\leq1\right\}$$
- 对 $2$ 的证明思路为我们需要同时证明 $\operatorname{D}_{\text{var}}(P,Q)\leq\sup_f\left(\int_\mathcal{R}f(x)P(dx)-\int_\mathcal{R}f(x)Q(dx)\right)$ 以及 $\operatorname{D}_{\text{var}}(P,Q)\geq\sup_f\left(\int_\mathcal{R}f(x)P(dx)-\int_\mathcal{R}f(x)Q(dx)\right)$
- 对 $3$ 的证明思路与 $2$ 的证明思路相似，但是在证明 $\geq$ 的过程中，我们需要根据下确界的定义（因为 $f$ 是有界的，有下界必定存在下确界）从而构造一个新的函数 $f_\varepsilon(x)=\frac{f(x)-f(y)+\varepsilon}{1+\varepsilon}\in[0,1]$ where $\exists y,f(y)\geq \inf f+\varepsilon$.

**Warning**
If $P_n\rightarrow P$, $P_n\neq P$, (e.g. $P_n=P+\frac{1}{n}$)
$$
\operatorname{D}_{\text{var}}(\delta_p,\delta_{p_n})=1\nrightarrow 0
$$
即弱收敛并不意味着变分距离收敛。

# Divergence

Divergence衡量的是：一个分布在多大程度上“不像”另一个分布；或者说如果我用分布 $Q$ 来近似真实的分布 $P$ 我会犯多少错。

## Kullback-Leibler (KL) Divergence

$$
\text{Discrete case:}\operatorname{KL}(P\|Q)=\sum_{x\in\mathcal{R}}\left[(\log\frac{P(x)}{Q(x)})P(x)\right]
$$

## $\alpha$ - Divergence

>[!note] Definition
>Let $\alpha:(0,+\infty)\rightarrow [0,+\infty)$ be a non-negative convex function such that $\alpha(1)=0$, $\alpha(t)>0,t\neq1$. Let $P,Q$ be two probability distributions on some space $\mathcal{R}$, and $\mu$ a measure on $\mathcal{R}$ such that $P,Q<<\mu$ with density $\varphi_P,\varphi_Q$, then the $\alpha$-divergence between $P$ and $Q$ is defined by
>$$
>\operatorname{D}_\alpha(P\|Q)=\int_{\mathcal{R}}\varphi_Q(x)\alpha(\frac{\varphi_P(x)}{\varphi_Q(x)})d\mu
>$$
>with the convention
>$$
>\alpha(0)=\lim_{t\rightarrow0}\alpha(t)\qquad 0\alpha(f/0)=f\lim_{t\rightarrow0}\alpha^*(t)
>$$
>where $\alpha^*(t)=t\alpha(1/t)$.
- $\operatorname{D}_{\alpha^*}(P\|Q)=\operatorname{D}_{\alpha}(Q\|P)$ (证明只需代入后化简即可)
- This divergence is, in general, **not symmetric** $\operatorname{D}_\alpha(P\|Q)\neq\operatorname{D}_\alpha(Q\|P)$ (因为“错在谁是不一样的”), nor does it satisfy the triangular inequality.
	- if we consider $\beta=\frac{\alpha^*+\alpha}{2}$, then $\operatorname{D}_\beta$ is symmetric$$\operatorname{D}_\beta(P\|Q)=\frac{1}{2}(\operatorname{D}_\alpha(P\|Q)+\operatorname{D}_\alpha(Q\|P))$$
	- Kafka's condition: if we consider $h(t)=\frac{|t^s-1|^{1/s}}{\alpha(t)}$ which is well defined on $(0,+\infty)$, non-increasing on $(0,1)$ and continuous at $t=1$, then $\operatorname{D}_\alpha(P\|Q)^s$ is a symmetric and satisfies the triangle inequality.
- if we let $\alpha(t)=|t^s-1|^s$, then we can get $\operatorname{D}_\alpha(P\|Q)=\int|\varphi_P^\alpha-\varphi_Q^\alpha|^{1/\alpha}d\mu$ 
	- when $s=1$, $\operatorname{D}_\alpha(P\|Q)=2\operatorname{D}_{\text{var}}(P,Q)$.
	- when $s=\frac{1}{2}$, we get the Hellinger distance $\int(\sqrt{\varphi_P}-\sqrt{\varphi_Q})^2d\mu$
- ![[Jensen-Shannon divergence.png]]
$$
\operatorname{D}_{\varphi_1}(P\|Q)=\operatorname{KL}(P\|\frac{P+Q}{2})+\operatorname{KL}(Q\|\frac{P+Q}{2}) 
$$
is Jensen-Shannon divergence.

# Monge-Kantorovich Distance

Monge-Kantorovich 距离衡量的是：把一个概率分布“最优地搬运”成另一个分布，所需要付出的最小“搬运”成本。

>[!note] Definition
>We define a transportation cost $\rho(x,y):\mathcal{R}\times\mathcal{R}\rightarrow +\infty$, for moving a unit of mass from $x$ to $y$. Then to evaluate the minimum total cost needed to transform the distribution $P$ into $Q$, we define Monge-Kantorovich distance as
>$$
>\operatorname{D}_{MK}(P,Q)=\inf_{\pi\in M(P,Q)}\int_{\mathcal{R}\times\mathcal{R}}\rho(x,y)\pi(dx,dy)
>$$ 
>where $M(P,Q)$ is the set of all joint distributions on $\mathcal{R}\times\mathcal{R}$ whose first marginal is $P$ and second marginal $Q$.

## Special Case

$\exists g:\mathcal{R}\rightarrow\mathcal{R}$, earth at $x$ is moved to $g(x)$. Then we can get the cost (Monge Problem)
$$
\min_g\int_\mathcal{R}\rho(x,g(x))P(dx)
$$
such that $\forall B, P(g^{-1}(B))=Q(B)$, i.e. $g\# P=Q$. Then we have 
$$
\mu(A\times B)=P(A\cap g^{-1}(B))\Leftrightarrow \forall h,\int h(x,y)\mu(dx,dy)=\int h(x,g(x))P(dx)
$$
Let $h=\rho$, then Monge cost = Kantorovich cost for this $\mu$. And the marginal is
$$
\mu(A\times\mathcal{R})=P(A)\quad \mu(\mathcal{R}\times B)=Q(B)=P(g^{-1}(B))
$$

**Theorem:** If $\exists \alpha$, such that $\rho^{1/\alpha}$ is a distance on $\mathcal{R}$, then $\operatorname{D}_{MK}^{1/\alpha}$ is a distance on the space of probability measures on $\mathcal{R}$.
- When $\alpha=1$, i.e. $\rho$ is a distance. We call a function $h:\mathcal{R}\rightarrow\mathbb{R}$ is $\rho$-distance if $\forall x,y,|h(x)-h(y)|\leq \rho(x,y)$. Then we define$$\operatorname{D}^*_\rho(P,Q)=\sup\left\{\int fdP-\int fdQ:f\text{ is }\rho\text{ -distance}\right\}
$$And we have **theorem (Kantorovich-Rubinstein)**$$
\operatorname{D}_{MK}(P,Q)=\operatorname{D}^*_\rho(P,Q)$$
If $\mathcal{R}=\mathbb{R}^d$, take $\rho(x,y)=|x-y|^p$, we then get $L^p$ MK distance.

## Maximum Mean Discrepancy

We consider $\mathcal{F}:$ any set of function $f:\mathcal{R}\rightarrow\mathbb{R}$. We define maximum mean discrepancy
$$
\operatorname{D}_\mathcal{F}^*(P,Q)=\sup\{\int fdP-\int fdQ:f\in\mathcal{F}\}
$$
