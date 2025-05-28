#MonteCarlo 
# Bias-Variance Tradeoff

Consider the model
$$
z\overset{\text{model}}{=}h(\boldsymbol{\theta},x)+\nu
$$
where $z$ is the actual scalar output, $h(\boldsymbol{\theta},x)$ is a regression function, $\boldsymbol{\theta}$ is regression coefficient vector, $x$ is the input vector, $\nu$ is the noise term that may or may not have mean zero.

## Conditional Mean-Squared Error (MSE)

MSE is a natural measure of effectiveness of the regression function. We define it as
$$
E\left[ (h(\theta, x) - z)^2 \mid x \right] = \underbrace{E\left[ (z - E(z \mid x))^2 \mid x \right]}_{\text{process variance (nonmodel)}} 

+ \underbrace{[h(\theta, x) - E(z \mid x)]^2}_{\text{model error}}

$$

The MSE for a regression prediction can be decomposed into a part due to the inherent variability of the process (i.e., $E\left[ (z - E(z \mid x))^2 \mid x \right]$)a part due to the error in the model at a specified $\theta$ (i.e., $[h(\theta, x) - E(z \mid x)]^2$)
- $E\left[ (z - E(z \mid x))^2 \mid x \right]$: not depend on the model, it simply reflects the conditional variance of the true process.
- $[h(\theta, x) - E(z \mid x)]^2$: directly related to the model, we can reduce it by choosing a better model.

One can also see [[Frequency Statistics#ANOVA Analysis|ANOVA in SLR]] or [[Frequency Statistics#ANOVA Table for MLR|ANOVA in MLR]].

## Bias-Variance Tradeoff

$$
E \left[ \left( h(\hat{\theta}_n, x) - E(z \mid x) \right)^2 \mid x \right]= \underbrace{E \left[ \left( h(\hat{\theta}_n, x) - E(h(\hat{\theta}_n, x) \mid x) \right)^2 \mid x \right]}_{\text{variance at } x} 

+ \underbrace{\left[ E(h(\hat{\theta}_n, x) \mid x) - E(z \mid x) \right]^2}_{(\text{bias at } x)^2}.
$$

An unbiased estimator is one with $E\left[h(\hat{\theta}_n,x)|x\right]=E[z|x]$.

As a final overall assessment of contributions toward the model MSE, we have
$$
\begin{array}{rl}
\text{MSE}_{\text{overall}} &= E_x \left[ E \left\{ \left( h(\hat{\theta}_n, x) - E(z \mid x) \right)^2 \mid x \right\} \right]\\

&= E_x \left[ \underbrace{\text{variance at } x}_{\text{variance}} + \underbrace{(\text{bias at } x)^2}_{\text{bias}^2} \right]\\


&= \overline{\text{variance}} + \overline{\text{bias}^2}
\end{array}
$$

>[!question] Is unbias always best?
>We consider $z_1,\cdots,z_n$ are i.i.d. random variable with mean $\mu$ and variance $\sigma^2$. We want to estimate $\mu$.
>
>Consider the estimate $r\bar{z}$, where $r\in(0,1]$. Then we can calculate MSE
>$$
>MSE=(r-1)^2\mu^2+r^2\frac{\sigma^2}{n}
>$$
>We optimize MSE w.r.t. $r$. By taking first and second order derivative, we have
>$$
>\frac{\partial MSE}{\partial r}=2(r-1)\mu^2+2r\frac{\sigma^2}{n}=0\Rightarrow r=\frac{n\mu^2}{n\mu^2+\sigma^2}\qquad \frac{\partial^2 MSE}{\partial r^2}>0
>$$
>Then MSE has the minimum. If we know $\mu$, $r\bar{z}$ is better.
>This example means some biased estimators may have lower MSE. And **in practice, unbias may be not necessary.**


## Model Complexity  v.s.  Bias/Variance

$$
\begin{aligned}
\text{Simple Model}\Leftrightarrow \text{High bias/low variance}\\
\text{Complex Model}\Leftrightarrow \text{Low bias/high variance}
\end{aligned}
$$

![[complexity model vs bias:vairance.png]]

Four contributors to generalization
- Complexity of true process
- Quantity and quality of training data
- Architecture of mathematical model
- Training algorithm


## Double-Descent Phenomenon

**Background:** high-dimension, i.e. number of parameters $p\gg n$, number of samples. 

Double descent refers to case where test error **first decreases**, then **increases** as model complexity (e.g., number of parameters) increases, then **decreases again.**

Double-descent performance curve has been experimentally observed in diverse machine learning settings.

![[double-descent.png]]




# Model Selection

**Practical aim:** to pick a model that minimizes a criterion
$$
f_1(\text{fitting error from given data})+f_2(\text{model complexity})
$$
where $f_1$ and $f_2$ are increasing functions.

See [[Frequency Statistics#Model Selection|Model Selection Criterion]].

## Regularization

**Idea:** Standard batch least-squares loss function for estimating $\theta$:
$$
\hat{L}_n(\theta)=\frac{1}{2n}\sum_{k=1}^n(z_k-h(\theta,\mathbf{x}_k))^2
$$
will lear to over-parameterization with too many components in $\theta$. See [[Frequency Statistics#Extra Sum of Squares|Extra Sum of Squares]].

Our "Regularization" is to add another function as penalty term to least-squares function to **penalize large-dimensional $\theta$**.

Typical form with regularization is
$$
R_n(\theta)=\hat{L}_n(\theta)+f(p)
$$
where $f(\cdot)$ is some increasing function and $p=\dim(\theta)$.

### Sjöberg and Ljung Criterion

$$
R_n(\theta)=\hat{L}_n(\theta)+\eta\|\theta-\theta'\|^2
$$
where $\eta>0$ and $\theta'$ is fixed value of $\theta$ based on our prior.

For the penalty term, we have
$$
\|\theta-\theta'\|_{\dim(\theta)=p}-\|\theta-\theta'\|_{\dim(\theta)=p'}\geq(p-p')d^2>0
$$
where $p>p'$ and components of $\theta,\theta'$ differ by at least a fix amount $d$.

## Cross-Validation
















