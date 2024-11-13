# 3D Gaussian Splatting

贡献：提出了新的3D显式几何表示方法(3DGS)

特点：渲染速度快，渲染质量高

应用：实时渲染

## pipeline

![](D:\notebook\论文方法论\picture\3dgs_pipeline.png)

**一、初始化点云**

通过现有的 **Structure From Motion(SFM)** 方法，采用开源软件 COLMAP，以图像数据集为输入，输出得到表示目标场景的点云数据

> COLMAP 源码地址：[colmap/colmap: COLMAP - Structure-from-Motion and Multi-View Stereo](https://github.com/colmap/colmap)
>
> COLMAP 论文
>
> 1.  Structure-from-Motion Revisited
> 2.  Pixelwise View Selection for Unstructured Multi-View Stereo

**二、生成 3D Gaussian**

属性

- mean $\mu$，点云中每个点的位置，3D Gaussian以点云中的点为中心 $\mu \in \mathbb{R}^{3 \times 1} $

- color：表示颜色。directional appearance

  球谐系数(Spherical Harmonic coefficient) $f_k \in \mathbb{R}^{3 \times 16}$，view-dependent color $c_k \in \mathbb{R}^{3 \times 1}$

- opacity $\alpha$：表示透明度

- $\Sigma$，$3 \times 3$的协方差矩阵(半正定矩阵)。$\Sigma = R S S^T R^T$ , 表示旋转和缩放。用于控制3D Gaussian的形状

  > 协方差矩阵的初始化：各向同性高斯，坐标轴等于 到三个最近点距离的平均值

定义：$$G(X) = e^{-\frac{1}{2}(x)^T\sum^{-1}(x)}$$

3D Gaussian的几何形状为椭球体

选择 3D Gaussian 作为 scene representation的理由：

1. differentiable volumetric representation
2. preserves properties of volumetric rendering for optimization
3. rasterized very efficiently by projection them to 2D, apply $$\alpha -blend $$, sorted

3DGS高效率的关键

1. tile-based rasterizer
   * a-blending of anisotropic splat. visibility order(fast sorting)
2. backward pass by tracking accumulated $\alpha$ 



特性：

- 各向异性(Anisotropy)：不同观测角度下，物体的性质不同。

**三、投影**

渲染期间，像素的颜色由特定视角下 3D Gaussian 投影到 屏幕上决定，投影公式为：
$$
\Sigma^{'} = JW\Sigma W^TJ^T
$$
其中 $\Sigma^{'}$是投影到屏幕上的 高斯椭圆体的协方差矩阵，$J$是投影变换(仿射近似)的雅可比矩阵，$W$是视口变换

$\Sigma^{'}$是View Space 的协方差矩阵，$\Sigma$是 world space 的协方差矩阵



**Differentiable Tile Rasterizer**

1. 将屏幕分为16 * 16个 tile
2. cull 3D Gaussian。剔除在99%的视锥体置信区间之外的3D Gaussian，剔除 means 接近 近平面和远平面
3. 3D Gaussian投影到特定tile区域
4. 建立数据对（Tile ID, view space depth），一个3D Gaussian每投影到一个tile区域，就建立一个数据对。对数据对进行分类（相同Tile 分在一起）
5. 根据depth进行排序
6. 光栅化：对每个Tile建立一个TLB，对每个像素，根据排序累计$\alpha$、color，确定每像素颜色。终止条件：$\alpha$饱和

$\alpha$ blending

作用：减少由于离散3D Gaussian造成的 hole 走样问题，



## 优化

3DGS算法思想：通过不断优化3D Gaussian参数来创建辐射场

**位置**

优化算法：standard exponential decay scheduling 

**协方差矩阵 $$ \Sigma $$**

3D Gaussian 投影位置错误，错误表示 3D scene

优化算法：

- 随机梯度下降算法
- exponential activation function(scale of covariance)



**透明度 $$ \alpha$$**

alpha检测：沿光线计算对应像素值，根据通过的透明度(光照强度衰减)迭代

优化算法：sigmoid activation function. 保证限制在[0, 1)，获取平滑的梯度

**拟合**

3D Gaussian的分解(split)与聚集(clone)

Adaptive Control：

重点 region：

- under-reconstruction(region that missing geometric feature, too small)：clone gaussians
- over-reconstruction(gaussian cover large areas in the scene, too large)：split into two smaller one

共同点：have large view-space positional gradients

增加、移除、修改 3D Gaussian

移除 透明度低于阈值

**性能**

- fast differentiable rasterizer

  

## 局限性

- 对于 not well observed 的区域有 artifact

- memory cost is high