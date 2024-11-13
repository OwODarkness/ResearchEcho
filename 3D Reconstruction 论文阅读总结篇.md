# 论文阅读总结

## 研究方向

### Novel 3D Representation

**NeRF(Neural Radiance Field 神经辐射场)**

**3DGS(3D Gaussian Splatting)**

### 可微渲染(differentiable rendering)

**正向渲染**

$y = f(x)$	以场景参数为输入，得到2D渲染图像

场景参数【几何(geometry)、相机(camera)、材质(material)、光照(lighting)】

**逆向渲染**

由对应某场景的2D渲染图像集为输入，得到场景参数

**可微渲染**

$\frac{\partial y}{\partial x}$

可微渲染——逆向渲染，梯度下降，反向传播

可微光栅化

欧拉视角：固定像素位置p的颜色变化

拉格朗日视角：3D RGB颜色变化，屏幕投影位置变换

## 阅读清单

### 已读

**论文**

- Mildenhall, Ben, et al. "Nerf: Representing scenes as neural radiance fields for view synthesis." *Communications of the ACM* 65.1 (2021): 99-106
- Wu, Tong, et al. "Recent advances in 3d gaussian splatting." *Computational Visual Media* 10.4 (2024): 613-642.【Survey】
- Kerbl, Bernhard, et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Trans. Graph.* 42.4 (2023): 139-1.

**书籍**

- fundamentals-of-computer-graphics-4th

**其他**

* An Introduction to BRDF-Based Lighting	NVIDIA Corporation Chris Wynn
* 

### 未读

