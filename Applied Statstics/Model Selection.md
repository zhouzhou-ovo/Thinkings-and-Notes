

Model Selection = Variable Selection = Subset Selection

**Parsimony Principle:** Model with fewest number of predictors, but still giving "MOST information".

### Coefficient of Multiple Determination $R_p^2$

$p$: number of parameters with $p-1$ predictors
$$
R_p^2=1-\frac{SSE}{SST_o}
$$

>[!question]
>Why $R_p^2$ is not a good measure?
>
**Answer:**
>
>As $p \uparrow$ , $SSE \downarrow$ $\Rightarrow$ $R_p^2 \uparrow$ as we add predictors $\Rightarrow$ Overfitting

![[Rp.jpg]]

### Adjusted $R^2$: $R^2_{adj,p}$

$$
R^2_{adj,p}=1-\frac{MSE_p}{MST_o}=1-\frac{SSE}{SST_o}\frac{n-1}{n-p}
$$
where $n-p$ is penalty term.

As $p\uparrow$, $SSE_p\downarrow$, $\frac{1}{n-p}\uparrow$ $\Rightarrow$ balance between $\downarrow$ in SSE and penalty due to extra variable.

Higher $R^2_{adj,p}$ , better model.

![[adjusted R.jpg]]

### Mallow's $C_p$ (measure of predictive quality of the model)

$C_p=$ estimator of a R.V. $\Gamma_p$ (gamma: total mean of squared error)

$\hat Y_i=$ estimated mean of observation $i$ by a model

$\mu_i=$ true population mean of obs'n $i$ in the true model

$\hat Y_i-\mu_i=(\hat Y_i-E[\hat Y_i])+(E[\hat Y_i]-\mu_i)$ where the first term is error component and the second term is bias.

$E[(\hat Y_i-\mu_i)^2]=Var(\hat Y_i)+(E\hat Y_i-\mu_i)^2$ 

Then we define
$$
\Gamma_p=\frac{1}{\sigma^2}\sum_i E(\hat Y_i-\mu_i)^2=\frac{1}{\sigma^2}\left[\sum_i Var(\hat Y_i)+\sum_i(E\hat Y_i-\mu_i)^2 \right]
$$
which measures **how far $\hat Y_i$ is from $\mu_i$ on average.**

>[!Note] Fact
>Mallow's $C_p=\hat \Gamma_p=\frac{SSE_p}{MSE_p}-(n-2p)=\frac{SSE_p}{MSE(X_1,X_2,\cdots,X_{p-1})}-(n-2p)$
>
>where $SSE_p$ is $SSE$ with $p-1$ predictors and $MSE(X_1,X_2,\cdots,X_{p-1})$ is $MSE$ with all predictors.

smaller $C_p$, better model.

As $p\uparrow$, $SSE_p\downarrow$, but $2p-n \uparrow$ 

We look for model with smallest $C_p$ and the values $C_p\approx p$ .

>[!Note] Fact
>$E[C_p]\approx p$, when unbiased.

![[Cp.jpg]]

### $AIC_p=$ Akaike Information Criterion

$$
AIC_p\overset{\text{def}}{=}n\log SSE_p-n\log n+2p
$$

As $p\uparrow,SSE_p\downarrow,2p\uparrow\Rightarrow$ balanced.

Smaller $AIC_p$, better model.

**Warning:** Different packages have different definition of $AIC$, but they all agree on $\underset{p}{\arg\min} AIC_p$.

### $SBC_p=BIC_p$

Schwartz Bayesian Criteria = Bayesian Information Criteria

$$
\begin{array}{rl}
SBC_p\overset{\text{def}}{=}n\log SSE_p - n\log n+(\log n)p
\end{array}
$$
>[! question]
>$SBC_p$ vs $AIC$, which one will choose a smaller model?
>
>**Ans:**
>$$
>\begin{array}{rl}
>\text{when }\log n>2(n>8)&\Rightarrow SBC_P\text{ more penalty on large model}\\
>&\Rightarrow SBC_p \text{ tend to be more parsimonious, i.e. choose smaller model}
>\end{array}
>$$

| Criteria | $SBC_p$               | $AIC_p$                                        |
| -------- | --------------------- | ---------------------------------------------- |
| Penalty  | $(\log n)p>2p$        | $2p$                                           |
| Model    | choose the true model | choose a model that contains<br>the true model |

### $PRESS_p=$ predicted sum of squares

$$
PRESS_p\overset{\text{def}}{=}\sum_{i=1}^n(Y_i-\hat{Y}_{i(i)})^2=\sum_{i=1}^n d_i^2
$$
where $d_i$ is [[Multiple Linear Regression(MLR)#Deleted Residuals|deleted residual]].

We look for model with smallest $PRESS_p$, i.e. model that are best at predicting "unseen" data.

### Automatic Procedures (for many predictors)

#### Best Subsets

See six criteria above
- Pro: can give you several truly "best" models
- Cons: Too expensive with many predictors.
#### Forward Stepwise

- Step 1: $Y\thicksim X_1(t_1^*),\cdots,Y\thicksim X_{P-1}(t_{P-1}^*)$ 
	- If $\max|t_k^*|>t_{\text{thresh}}^*\Rightarrow$ keep $X_k$ (e.g. $Y\thicksim X_7$)
	- If $\max |t_k^*|<t_{\text{thresh}}^*\Rightarrow$ stop (none of the predictors have significant relationship)
- Step 2: $Y\thicksim X_7+X_1(t_{7,1}^*),\cdots,Y\thicksim X_7+X_{P-1}(t_{7,P-1}^*)$
	- If $\max |t^*_{7,k}|>t^*_{\text{thresh}}\Rightarrow$ keep $X_k$
	- If $\max |t^*_{7,k}|<t^*_{thresh}\Rightarrow$ stop ($Y\thicksim X_7$ is the best model)
- Step 3: Look at marginal $t^*$ for previously added predictors (e.g. $Y\thicksim X_7+X_3$, check $t^*$ for $X_7$)
	- If too low $\Rightarrow$ drop $X_7$
	- If not $\Rightarrow$ keep $X_7$
	- Then repeat (go to Step 2)

#### Forward Selection

Skip Step 3.

---