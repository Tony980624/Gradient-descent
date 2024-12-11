# Gradient-descent
理解梯度下降的原理

假设一个函数y,它和x的真实关系是: $y=x^2 + 2x+8$, 但是由于噪声（数据中其他和y相关变量的影响）的存在,使得难以直接知道y和x之间函数的参数

```
import numpy as np
import matplotlib.pyplot as plt
x =  np.arange(-10,10,0.1)
y = x**2 + 2*x+8
plt.plot(x,y)
```

