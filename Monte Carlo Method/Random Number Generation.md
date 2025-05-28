#MonteCarlo
# Linear Congruential Generators

Consider recursive formula
$$
X_{t+1}=aX_t+c\,(\text{mod }m)
$$
where $X_0$ is called the seed. And we can find $X_t\in\{0,1,\cdots,m-1\}$.

The quantities $U_t=\frac{X_t}{m}$ is called **pseudorandom numbers**(伪随机数). The sequence will repeat itself after at most $m$ steps and will therefore be periodic, with a period not exceeding $m$.

# Multiple Recursive Generators

If we consider k-dimensional state vectors $\mathbf{X}_t=(X_{t-k+1},\cdots,X_t)^T,t=0,1,2,\cdots$ and rewrite the formula as
$$
X_t=(a_1X_{t-1}+\cdots+a_kX_{t-k})\,\text{mod }m,\quad t=k,k+1,\cdots
$$

We give seed $\mathbf{X}_0=(X_{-k+1},\cdots,X_0)$. The maximum period length for this generator is $m^k-1$. And when $m$ is a large integer, the output stream of random numbers is obtained via $U_t=X_t/m$.

MRGs with very large periods can be implemented efficiently by combining several smaller period MRGs — yielding **combined multiple-recursive generators**.

# Modulo 2 Linear Generators

**Idea:** binary operations are in general faster than floating point operations.

Consider $\mathbf{X}_t=(X_{t,1},\cdots,X_{t,k})^T,\mathbf{Y}_t=(Y_{t,1},\cdots,Y_{t,w})^T$, we can get the random number $U_t\in(0,1)$ by bitwise decimation as follows:
![[M2G.png]]
where $A,B$ are $k\times k,w\times k$ binary matrix, and all operations are performed modulo 2.