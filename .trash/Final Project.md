![[Pasted image 20241103183845.png]]
![[Pasted image 20241103200200.png]]
![[Pasted image 20241104101211.png]]
![[Pasted image 20241104101431.png]]
![[Pasted image 20241104101452.png]]
![[Pasted image 20241104101616.png]]
![[Pasted image 20241104101721.png]]
![[Pasted image 20241104101801.png]]
![[Pasted image 20241104101817.png]]
![[Pasted image 20241104101948.png]]
![[Pasted image 20241104102304.png]]
![[Pasted image 20241104102419.png]]
![[Pasted image 20241104102717.png]]

## Presentation

数据描述：
- 我们发现 residual.sugar, free.sulfur.dioxide, total.sulfur.dioxide, alcohol 具有更大的方差，同时其他的feature的方差均较小，说明上述四个方差较大的feature可能在影响white wine的质量中具有重要的作用，也就是说我们未来建立的模型中很大可能会包括这些关键变量
- 我们同样也发现数据集中white wine的质量评分主要集中在5和6，这意味着我们的数据集并不是一个分类较为平衡的数据集。所以我们为了保证划分高质量与低质量后的分类变量是均衡分类，故选择起quality>=6的white wine为高质量，低于6的是低质量
- 我们还发现density和alcohol与其他features之前具有较高的correlation coefficients，这说明density和alcohol与其他features之间可能存在multicollinearity，在未来建立的模型中我们可能会考虑不加入这两个features或者保留这两个变量，删除一些其他的features
- 我们还发现不同feature之间存在较大的数量级差异，因此为保证模型拟合的准确度，我们需要对原数据集进行标准化

Methods

For our white wine dataset, we plan to use logistic regression to classify wine quality. And the regression model is shown in the slide. The dimension of coefficient is 12 by 1.
our classification rules are:
if prob. of Q equals to 1 is larger or equal to 0.5, then let Q=1 and this white wine is high quality;
if prob less than 0.5, then Q=0 and the quality is low.

We use Bayesian Analysis to estimate the model coefficients. We first choose uninformative prior, normal distribution with mean 0 and variance 100. And we can also calculate the sampling model. Then for model fitting, we use metropolis to generate coefficients samples. And for model selection, we use metropolis-hasting to choose the best model.

Preliminary Findings

Training set selection: We randomly select 70% of the data in the original dataset as the training set.

Fitting: Then we use all features to fit the model. And Figure 4 is our model summary. We can find feature fixed.acidity, citric.acid, chlorides and total.sulfur.dioxide are not significant. So there may exists multicollinearity in the model. 

Model Selection: Next we take model selection to get best model. We remain volatile.acidity, residual.sugar, free.sulfur.dioxide, total.sulfur.dioxide, density, pH, sulphates and alcohol. But we still find total.sulfur.dioxide is still not significant.

Multicollinearity: Therefore, we use VIF to test Multicollinearity in the model. We find VIF of residual.sugar and density are larger than 10. So we drop density and refit the model. And now all features are significant.

Influence and Outliers:
Next step we consider influential obs'n and outliers. We first draw the plot of Cook distance of model. We find there exists many influential observations. We need to find which are outliers w.r.t y or x. We then get the ESR of the model. We find most points are between two critical values but there are two points out of range. So they are outliers w.r.t y. We then delete these 2 outliers and refit the model.

And the final model summary shows that all feature are significant. Next we need to consider model fitting effect. We take Hosmer-Lemeshow goodness of fit test and get ROC curve.
The test result is p-value equals to 0.2824 larger than 0.05, which means we accept null hypothesis the model fits well. And the shape of ROC curve and AUC values equals to 0.8 also show our model fits well.

These are our all preliminary findings. And our next plan is to use Bayesian approach to finish model fitting and selection, and use testing set to predict and consider 
