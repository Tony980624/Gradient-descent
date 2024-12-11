理解梯度下降的原理

假设一个函数y,它和x的真实关系是: $y=x^2 + 2x+8$, 这里函数右三个系数，分别是1，2，和常数8.怎么求解这三个参数？

# 定义损失方程

$Loss = (y_{real}-y_{pred})^2 = (y_{real}-ax^2-bx-c)^2$

**链式法则**对参数分别求导:

$\frac{\partial{L}}{\partial{a}} = -\frac{2}{N}\sum_{i=1}^Nx^2(y_{real}-ax^2-bx-c) = -\frac{2}{N}\sum_{i=1}^Nx^2(y_{real}-y_{pred})$

$\frac{\partial{L}}{\partial{b}} = -\frac{2}{N}\sum_{i=1}^Nx(y_{real}-ax^2-bx-c) = -\frac{2}{N}\sum_{i=1}^Nx(y_{real}-y_{pred}c)$

$\frac{\partial{L}}{\partial{c}} = -\frac{2}{N}\sum_{i=1}^N(y_{real}-ax^2-bx-c) = -\frac{2}{N}\sum_{i=1}^N(y_{real}-y_{pred}) $

# 生成数据

```
import numpy as np
import matplotlib.pyplot as plt
x =  np.arange(-10,10,0.1)
# 加入随机噪音
y = x**2 + 2*x+8 +np.random.normal(0,1,size=x.shape)
plt.plot(x,y)
```
![da](https://github.com/Tony980624/Gradient-descent/blob/main/output.png)

# 梯度下降函数

```
import pandas as pd
def gradient_descent(initial_theta,learning_rate,iterations):
    a=  initial_theta[0]
    b = initial_theta[1]
    c= initial_theta[2]
    list = []
    losses = []
    for i in range(iterations):
        y_pred = a*x**2+b*x+c
        #当前参数下的总误差
        MSE = np.mean((y_pred - y_real)**2)
        losses.append(MSE)
        error = y_real - y_pred
        # a的梯度
        gradient_a = -2*np.mean(error*x**2)
        gradient_b = -2*np.mean(error*x)
        gradient_c = -2*np.mean(error)
        # 根据学习率更新参数
        a = a -learning_rate*gradient_a
        b = b- learning_rate*gradient_b
        c = c-learning_rate*gradient_c
        list.append([a,b,c])
    return list,losses
```

# 查看损失的变化

![da](https://github.com/Tony980624/Gradient-descent/blob/main/output2.png)

损失下降的很快

# 查看三个参数的迭代变化

![da](https://github.com/Tony980624/Gradient-descent/blob/main/output3.png)

看到参数无线逼近正确答案a=1,b=2,c=8
