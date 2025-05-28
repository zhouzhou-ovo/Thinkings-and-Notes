
Objective Bayesian Analysts prefer priors which "do not strongly influence" the posterior distribution. Such a prior is called an **uninformative prior**.

# Flat Prior

Consider $p(\theta)=1,\theta\in[0,1]$.

- The good
	- uninformative, all values of $\theta$ are equally likely.
	- can make it improper
		We can let $\theta\in\Theta=(0,+\infty)$ or $\Theta=(-\infty,+\infty)$.
		Although $\int_{\Theta}p(\theta)d\theta\rightarrow \infty$ , but it will be good if our posterior distribution is proper.
	- Posterior is proportional to the likelihood, i.e.
	$$p(\theta|y_{obs})\propto p(y_{obs}|\theta)p(\theta)=p(y_{obs}|\theta)=L(\theta|y_{obs})$$
	which also means MLE $=$ MAP(Maximum a posteriori).

- The bad
	- If it is improper, $\int p(\theta)d\theta=\infty$, it assigns disproportinate weight to the larger values of $\theta$.(也就是说$\theta$会具有不等的权重)
	- The 'flat' stops being flat after the transformation: $\tau=g(\theta)$

>[!example] Unflat (Binomial Model)
>if $\theta\sim Beta(1,1)=Unif[0,1]$, $\tau=\log\frac{\theta}{1-\theta}\overset{not}{\sim}Unif$

>[!example] Binomial Model $Y|\theta\sim Bin(n,\theta)$
>- $\theta\sim Beta(1,1)$ is uniform for $\theta\in[0,1]$
>- $\theta\sim Beta(0,0)$ is uniform for log-odds $\tau=\log\frac{\theta}{1-\theta}$ (improper)
>- $\theta\sim Beta(1,-1)$ is uniform for odds $\tau=\frac{\theta}{1-\theta}$ (improper)

# Jeffreys' Prior

>[!note] Jeffreys' Prior
>$$p_J(\theta)\propto \sqrt{I(\theta)}$$
>where$$\quad I(\theta)=-E_y\left[\frac{\partial^2\log p(y|\theta)}{\partial\theta^2}\vert\theta\right]=-E_y[\frac{\partial^2\log L}{\partial \theta^2}\vert\theta]$$
>- Property: transformation invariant (i.e. "uniformed with respect to transformations")
>	$$\text{If }p(\theta)\propto\sqrt{I(\theta)},\theta\rightarrow\phi\Rightarrow p(\phi)\propto\sqrt{I(\phi)}$$

**Proof**:
	First, we prove that $I(\theta)=-E_y\left[\frac{\partial^2\log p(y|\theta)}{\partial\theta^2}\vert\theta\right]=E_y[\left(\frac{\partial\log L}{\partial \theta}\right)^2\vert\theta]$.$$
	\begin{array}{rl}
	E_y[\left(\frac{\partial\log L}{\partial \theta}\right)^2\vert\theta] &=\int \frac{\partial \log L}{\partial\theta}\frac{\partial \log L}{\partial\theta}p(y|\theta)dy\\
	&=\int \frac{1}{L}\frac{\partial L}{\partial\theta}\frac{\partial \log L}{\partial\theta}Ldy\\
	&=\int \frac{\partial L}{\partial\theta}\frac{\partial \log L}{\partial\theta}dy\\
	&=\int\frac{\partial}{\partial\theta}\left(L\frac{\partial \log L}{\partial\theta}\right)dy-\int L\frac{\partial^2 \log L}{\partial\theta^2}dy\\
	&=\int\frac{\partial}{\partial\theta}\left(L\frac{1}{L}\frac{\partial L}{\partial\theta}\right)dy+I(\theta)\\
	&=\frac{\partial^2}{\partial\theta^2}(\int Ldy)+I(\theta)\\
	&=I(\theta)
	\end{array}
	$$
	(从等式右侧向左侧证明，同时要注意积分与求导的交换性以及构造左侧形式)
	Second, we prove that $\text{If }p(\theta)\propto\sqrt{I(\theta)},\theta\rightarrow\phi\Rightarrow p(\phi)\propto\sqrt{I(\phi)}$. $$
	\begin{array}{rl}
	p(\phi)=p(\theta)\left\vert\frac{d\theta}{d\phi}\right\vert&\propto \sqrt{I(\theta)}\left\vert\frac{d\theta}{d\phi}\right\vert\\
	&=\sqrt{E_y[\left(\frac{\partial\log L}{\partial \theta}\right)^2\vert\theta](\frac{d\theta}{d\phi})^2}\\
	&=\sqrt{E_y[\left(\frac{\partial\log L}{\partial \theta}\frac{d\theta}{d\phi}\right)^2\vert\theta]}\\
	&=\sqrt{E_y[\left(\frac{\partial\log L}{d\phi}\right)^2\vert\phi]}=\sqrt{I(\phi)}
	\end{array}$$

- Jeffreys' Prior is uninformative prior because it is derived from Fisher information, which only depends on likelihood function and it minimizes the influence of subjective choices.
- Jeffreys' Prior can be proper or improper based on our model

>[!example] Jeffreys' Prior for Different Models
>Binomial Model
>$$
>p_J(\theta)\propto\theta^{-1/2}(1-\theta)^{-1/2}=Beta(1/2,1/2)
>$$
>(conjugate, **proper**, proper posterior)
>
>Poisson Model
>$$
>p_J(\theta)\propto\theta^{-1/2}=\lim_{b\rightarrow 0}Gamma(1/2,b)
>$$   
>(conjugate, **improper**, proper posterior)
