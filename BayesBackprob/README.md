## notes:
1. 用plain SGD训练，BBMlp收敛很快，而VanillaMLP和DropoutMLP的收敛却很慢，而且后两者的性能比BBMlp低。
2. 用SGD+momentum训练，BBMlp收敛速度与使用plain SGD时差别不大，但对VanillaMLP和DropoutMLP的收敛速度提升很多。同上者一样后两者的性能比BBMlp低。
3. 用Adam算法训练，VanillaMLP和DropoutMLP的收敛速度快过BBMlp，但对BBMlp的收敛速度影响不大。不过VanillaMLP和DropoutMLP的性能要优于BBMlp。
4. BBLayer的参数初始化对模型能否训练成功影响很大。

## results:
1. SGD+MO
![](https://github.com/yxue3357/MyResearchCodes/raw/master/BayesBackprob/sgd_mo.png)

注： 一些结果跟论文上的不太符合，不知道是否代码问题。