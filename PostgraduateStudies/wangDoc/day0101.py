import numpy as np
import matplotlib.pyplot as plt

# 生成 x 数据：0 到 2π
x = np.linspace(0, 2 * np.pi, 1000)
# 计算余弦值
y = np.cos(x)

# 绘图
plt.figure(figsize=(10, 4))
plt.plot(x, y, color='blue', linewidth=2, label=r'$y=\cos(x)$')

# 辅助设置
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('余弦函数 cos(x) 图像')
plt.xlabel('x (弧度)')
plt.ylabel('y')

plt.show()