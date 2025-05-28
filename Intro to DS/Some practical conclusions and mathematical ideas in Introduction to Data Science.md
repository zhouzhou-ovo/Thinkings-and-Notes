# 1. The distrib. of CDF as a RV

>[!Theorem 1]
> Let $X$ is a random variable with cumulative distribution function $F(X)$, then we have
> $$
> X=F^{-1}(U_{01})\text{, where }U_{01} \text{ is uniform between }0\text{ and }1
> $$
> that is 
> $$
> F(X)\thicksim U_{01}
> $$

这个定理理解的核心在于**概率与随机变量的取值是一一对应的**，那么 $F(X)$ 的概率与 $X$ 的概率也是一一对应的，这同样是我们证明的核心思想。现证明上述结论如下：
**Proof:**
Let $Y=F_X(X)$ , it is obviously that $Y\in (0,1)$. So when $Y \leq 0$, $F_Y(y)=0$; when $Y \geq 1$, $F_Y(y)=1$; when $0<Y<1$, we have
$$
\begin{array}{lcL}
F_Y(y)&=&P(Y\leq y)\\
&=&P(F_X(X)\leq y)\\
&=&P(X \leq F_X^{-1}(y))\\
&=& F_X(F_X^{-1}(y))\\
&=& y
\end{array}
$$
So we can get
$$
F_Y(y)=\begin{cases}
0,\quad y\leq0\\
y,\quad 0<y<1\\
1,\quad y\geq1
\end{cases}
$$
$$
Y\thicksim U(0,1)=U_{01}
$$
这个定理同时也说明了：**均匀分布在连续分布类中占有特殊的地位，任一连续随机变量都可以通过其分布函数与均匀分布随机变量发生关系。** 而根据这个定理，我们也得到了统计软件中生成随机数的方法：在确定分布后，优先生成均匀分布的随机数（这个在统计软件中都可以直接生成），然后利用均匀分布的随机数以及上述定理反过来计算其他分布的随机数。而获得各个随机分布的随机数也是随机模拟法（Monte Carlo Method）。

---
# 2. Histogram vs. Kernel Function

|         | 直方图（Histogram）           | 核密度估计（Kernel Function）                                                      |
| ------- | ------------------------ | --------------------------------------------------------------------------- |
| 原理      | 划分区间，使用频率近似概率            | 划分区间，使用核函数近似概率                                                              |
| 区间划分方法  | 人为选定起点、终点以及步长，然后对整个区间平均分 | 以数据点作为中心进行区间划分，中心差分 $[x_i-h,x_i+h]$                                         |
| 划分区间的特点 | 区间与区间互不相交                | 区间之间可能会出现块与块的相交                                                             |
| 缺点      | 图形并不光滑；结果受到端点以及步长的双重影响   | 唯一需要考虑的是步长选择带来的影响：较短的步长会导致不连续不平滑；较长的步长会导致过度平滑化（对渐进平均积分平方误差AMISE进行最小化得到最优步长） |
核密度估计法可以选择不同的核函数，如高斯核函数、Epanechnikov核、均匀核（Uniform Kernel）、三角核（Triangular Kernel）等函数，用来处理不同光滑化的需求。


对两者进行比较的相关文章：


[Histograms and kernel density estimation KDE 2 ]([Histograms and kernel density estimation KDE 2 | Biophysics and Beer (mglerner.github.io)](https://mglerner.github.io/posts/histograms-and-kernel-density-estimation-kde-2.html?p=28))


[An introduction to kernel density estimation]([An introduction to kernel density estimation (mvstat.net)](https://www.mvstat.net/tduong/research/seminars/seminar-2001-05/))

---




