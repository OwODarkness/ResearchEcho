# 3D重建基础知识篇

作者：owodarkness

## Introduction of 3D Reconstruction

三维重建是计算机视觉与计算机图形学的交叉研究方法，用于把现实的3D场景与物品在计算机中虚拟表示，重现于虚拟世界。目前，三维重建的发展方向正与AI不断进行融合。



在AI时代，许多事物值得被重新审视，AI 与 CG 会如何相互作用是个值得思考的问题。

在观看 Games001 公共网课期间，我听取了北京大学陈宝权教授的关于人工智能与计算机图形学关系的非常 inspired 的演说。 

结合演说与我个人的理解，AI 是在进行模仿游戏。评判 AI 好坏的标准或是AI对人类的模仿达到了什么样的水平，如著名的图灵测试。
人类从生活环境中学习，从实践中提升人的智慧。为训练出聪明的 AI，我们需要为它提供虚拟环境。因此，计算机图形学将在 AI 时代中承担构建虚拟环境，模拟虚拟世界的任务。



## 背景知识

### 数学

#### 线性代数

**坐标变换**

$y = Ax$，$A$是变换矩阵

- A视为向量组，当A是满秩的，则A内的向量线性无关。$x$和$y$的维度保持一致，
  - A可以张成一个完整的线性空间。张成表示A向量组的所有向量的任意的线性组合
  - $x$向量可以视为A各向量对应的缩放系数项，如$x_i$表示$A_i$的缩放系数
  - 若A是正交矩阵，A可视为一组基底，构 $x$是 $y$ 在A正交基下的表示

**齐次坐标**

齐次坐标空间是在欧式空间的基础上，额外引入一维。例如对于2D笛卡尔坐标$(x, y)$，其齐次坐标表示为：$(x, y, w)$。

以齐次坐标表示的点具有尺度不变性，即$(kx, ky, kw)$和$(x, y, w)$是等价的。

齐次坐标转为笛卡尔坐标：$(x, y, w) \rightarrow (x/w, y/w, 1) : (x/w, y/w)$ ，当w为0时，x, y无穷大，可视为无穷远的点

齐次坐标可用于区分以向量形式表示的点与向量，一般当 $w = 1$时表示点，$w = 0$表示向量

齐次坐标方便描述仿射变换，仿射变换包括：平移、旋转、伸缩、形变。在非齐次坐标表示下，平移变换只能用矩阵加法运算表示；而在齐次坐标表示下，平移信息可以存放在第N+1维，因此平移变换可以用矩阵乘法来表示

对于齐次坐标表示的空间，1-N维的列向量表示基

#### 自由度

完全确定该系统的状态所需要的最小的标量数

举例(以3维空间为例)

- 线段：线段由两个点确定，每个点有3个参数，因此线段的自由度为6
- 射线：射线由点和方向确定，点由3个参数表示，方向由两个参数表示（球面坐标），因此线段的自由度为5
- 直线：首先两个点确定线段有6个自由度，对于直线，两个点沿直线可以自由移动，减去两个自由度，所以直线有4个自由度



### 3D 几何表示方法

3D 几何表示方法分为隐式(implicit)表示和显式(explicit)表示，区别在于是否直接给出表示3D几何的数据，隐式不给出，显式给出

#### Implicit

- geometry surface：给出表示3D几何的隐式函数表达式，例如球面：$x^2 + y^2 + z^2 = 1$
- distance field：距离场

#### Explicit

- mesh：mesh一般由多个图元(primitive)组成，例如三角形。mesh给出所有表示图元的顶点以及顶点之间的关系

- point cloud：

  利用大量点来表示几何物体的表面

  - 每个点可以存储几何表示信息，例如 position, color, reflectance properties。
  - 每个点只会影响一个像素
  - dis-connected, 无需表示点与点之间的关系，更加简单的几何表示
  - 渲染存在缝隙，与密度相关

- voxel

### 视图变换

将3D object 从 3D坐标系 投影到 2D 坐标系下

坐标类型：物体空间(局部空间)，世界空间，摄像机空间，标准视图空间，屏幕空间

流程：

* 相机变换：根据 相机的pose(location + orientation) ，将物体从世界空间变换到摄像机空间
* 投影变换：将物体从摄像机空间投影到标准视图空间
* 视口变换：投影到2D平面



### 光栅化与光线追踪

#### 光栅化(rasterization)

目前图像显示设备大部分采用光栅显像方法，即屏幕为 $m \times n $的矩阵，矩阵内元素用采用RGB表示，一般称为像素

光栅化是将 3D object or scene representation 转换为 2D 并在 raster display上，即将几何信息转换成由光栅表示的图像

#### 光线追踪(ray tracing)

人眼成像原理：光源辐射，发出光线(光线是光的抽象表示方法)，光线打到物体表面，通过反射进入人眼

光线追踪是人眼人眼成像的逆过程，假设从摄像机出发出射线，当射线打到物体表面的同时与光线相交，则表示该处可以成像

光线追踪步骤：

1. 生成射线，射线由  $ p(t) =o + td$ 定义，其中$o$表示射线起点，$d$表示射线方向，$f(t)$表示射线在$t$时刻的分位点位置。

   观测(投影)方法一般分为正交与透视

   - 正交投影：射线起点为 viewport + offset，其中viewport 是摄像机位置，offset是像素点在 raster display 的位置偏移，射线方向为摄像机正视方向
   - 透视投影：射线起点 为 viewport，射线方向为 摄像机位置到目标像素点位置的方向向量

2. 射线与物体相交：已知射线 $p(t) =o + td$，在满足 $t>0$条件，求与射线相交的第一个物体的交点信息

   - Ray-Sphere Intersection（球形物体）：在已知射线$p(t) =o + td$ 和 球表面（球心$c =(x_c, y_c,z_c)$，半径$R$）的条件下， 沿射线，每间隔在$p$点($o+td$)进行球形检测。若满足 $f(p) = 0$，则发生相交
     $$
     (x -x_c)^2 + (y-y_c)^2+ (z-z_c)^2 = R^2\\
     (p-c) \cdot (p-c) - R^2 = 0\\
     (o+td -c) \cdot (o+td -c) - R^2 = 0\\
     d\cdot d t^2+ 2d\cdot(o-c) t+ (o-c)\dot(o-c) - R^2 = 0\\
     $$
     令 $A = d \cdot d$，$B = 2d\cdot (o-c)$，$C=(o-c) \cdot (o-c) - R^2$ 。原式化为：
     $$
     At^2 + Bt + C  =0
     $$
     求解 $t$，若 $t$有解且$t>0$，则表示相交

   - Ray-Triangle Intersection（三角形物体）：已知射线 $p(t) =o + td$ ，三角形顶点 $a, b,c$，若射线与三角形相交于点 $o + td$，此时 交点在三角形内，由重心坐标公式可得
     $$
      o + td = a + \beta(a-c)+\gamma(b-c)
     $$

     $$
     x_o + tx_d = x_a + \beta(x_a - x_b) + \gamma(x_a - x_c)&
     \beta(x_c - x_a) +\gamma(x_b - x_a)+ tx_d = x_a -x_o \\
     y_o + ty_d = y_a + \beta(y_a - y_b) + \gamma(y_a - y_c)&
     \beta(y_c - y_a) +\gamma(y_b - y_a)+ ty_d = y_a -y_o\\
     z_o + tz_d = z_a + \beta(z_a - z_b) + \gamma(z_a - z_c)&
     \beta(z_c - z_a) +\gamma(z_b - z_a)+ tz_d = z_a -z_o\\
     $$

     $$
     A =
     \begin{bmatrix}
     x_c - x_a & x_b - x_a & x_d\\
     y_c - y_a & y_b - y_a & y_d\\
     z_c - z_a & z_b - z_a & z_d
     \end{bmatrix}
     &&
     \begin{bmatrix}
     x_c - x_a & x_b - x_a & x_d\\
     y_c - y_a & y_b - y_a & y_d\\
     z_c - z_a & z_b - z_a & z_d
     \end{bmatrix}
     \begin{bmatrix}
     \beta\\
     \gamma\\
     t
     \end{bmatrix}
     =
     
     \begin{bmatrix}
     x_a - x_o\\
     y_a - y_o\\
     z_a - z_o
     \end{bmatrix}
     $$

     $$
     \beta = 
     \frac
     {
     \begin{vmatrix}
     x_a - x_o & x_b - x_a & x_d\\
     y_a - y_o & y_b - y_a & y_d\\
     z_a - z_o & z_b - z_a & z_d
     \end{vmatrix}
     }
     {A}
     
     &
     \gamma =
     \frac
     {
     \begin{vmatrix}
     x_c - x_a & x_a - x_o & x_d\\
     y_c - y_a & y_a - y_o & y_d\\
     z_c - z_a & z_a - z_o & z_d
     \end{vmatrix}
     }
     {A}
     &
     t = 
     \frac
     {
     \begin{vmatrix}
     x_c - x_a & x_b - x_a & x_a - x_o\\
     y_c - y_a & y_b - y_a & y_a - y_o\\
     z_c - z_a & z_b - z_a & z_a - z_o
     \end{vmatrix}
     }
     {A}
     $$

     

     求解 $\beta,\gamma,t$，若有解，且$\beta >0, \gamma>0, \beta+\gamma<1, t>0$

   - Ray-Polygon Intersection（多边形物体）:

3. 

### 辐射度量学

Radiant energy and Flux

- Radiant energy：辐射能量，单位焦耳

- Radiant flux：单位时间能量的发射、反射、传播和吸收的量，功率，通量，单位(Watt, lumen)
  $$
  \Phi = \frac{\mathrm{d}Q }{\mathrm{d}t}
  $$
  
- Radiant Intensity：点光源每单位**立体角**散发的能量
  $$
  I(\omega) = \frac{\mathrm{d\Phi}}{\mathrm{d}\omega}
  $$
  立体角：空间中的角 $\Omega = \frac{A}{r^2}$, 其中A为立体角对应的球的表面积
  $$
  \mathrm{d}\omega = \frac{\mathrm{d}A}{r^2}\\
  dA = (r\mathrm{d}\theta)(r\sin\theta)\mathrm{d}\phi = r^2\sin\theta \mathrm{d}\theta \mathrm{d}\phi\\
  \mathrm{d}\omega =\sin\theta \mathrm{d}\theta \mathrm{d}\phi
  $$
  
- Irradiance：单位面积入射光的能量，表面需要与光线垂直，单位面积的辐射通量。单位面积接受的总的辐射能量
  $$
  E(x) = \frac{\mathrm{d}\phi(x)}{\mathrm{d}A}
  $$
  
- Radiance：从某一个立体角辐射到单位面的能量。
  $$
  L(p, \omega) = \frac{\mathrm{d}^2\phi(p, \omega)}{\mathrm{d}\omega\mathrm{d}A\cos\theta }
  $$
  
- Radiance 和 Irradiance的关系
  $$
  L(p, \omega) = \frac{\mathrm{d}E(p)}{\mathrm{d}\omega\cos\theta }
  $$
  Irradiance：单位面积接收的总的能量

  Radiance：单位面积向某单位立体角方向发出的能力
  $$
  \mathrm{d}E(p, \omega) = L_i(p, \omega) \mathrm{d}\omega \cos \theta\\
  E(p) = \int_{H^2} L_i(p, \omega) \cos \theta \mathrm{d}\omega
  $$
  其中 $H^2$为半球区域

- 



Radiance 和 Irradiance 的差别：方向性

#### BRDF

属性：

- reciprocity：可逆性，入射和出射方向互换，则BRDF不变
- conservation of energy：出射的能量不会超过入射的能量

光与物体的交互类型：光的反射、光的吸收、光的透射
$$
light incident at surface = light reflected + light absorbed + light transmitted
$$

BRDF(Bidirectionala Reflectance  Distribution Function) 描述光从入射点出沿各方向$w_r$反射出的能量

$w$是单位立体角

入射光的irradiance与出射光的radiance的比率

the fuction of：

- **income(light) direction** $w_i$：$(\theta_i, \phi_i)$
- **outgoing(view) direction ** $w_r$：$(\phi_i, \theta_o)$
- wavelength(color)：$\lambda$
- positional variance：$ (u, v)$surface position parameterized

$$
\mathrm{BRDF}_{\lambda}(\theta_i, \phi_i, \theta_o, \phi_o, u, v)
$$


$$
f(w_i \rightarrow w_r) = \frac{\mathrm{d}L_r(w_r)}{\mathrm{d}E_i(w_i)} = \frac{L_r(w_r)}{L_i(w_i)\cos\theta_i\mathrm{d}w_i}
\begin{bmatrix}
\frac{1}{sr}
\end{bmatrix}
$$
其中 $sr^{-1}$是单位

渲染方程：
$$
L_r(p, \omega_r) = \int_{H^2} f_r(p, w_i \rightarrow w_r)L_i(p, \omega_i)\cos\theta_i\mathrm{d}\omega_i
$$

## 3D Reconstruction 研究方向

### novel 3D representation

* NeRF：神经辐射场3D隐式表示方法（神经网络+辐射场）

  提出时间：2020年

  > related paper：
  >
  > original paper：**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**

* 

1. COLMAP
2. NeRF
3. 3DGS



## 算法

**nearest neighborhood**

图像识别，寻找相似的图像，特征匹配，analysis data and features

特征：一事物用于区别另一事物的属性

neighborhood：共享同一特征的对象

识别特征，需要大量的数据，进行对比，

输出：概率

信息转为为向量，向量分布在空间中。利用每一个点寻找最近的点

数据向量化，数值化，放入数据库作为候选向量



每个对象由其向量作描述，其他与该对象的向量具有相似或相同的向量是候选对象，即neighborhood

KNN算法

根据已有的大量标记的数据集，对目标数据进行分类

标记：对数据下的定义

标记数据：supervised data	未标记数据：unsupervised data

k means? a representation of bounds 范围，how many object are you willing to consider, what is the family of the object

knn 表示目标数据对已有的数据label的匹配预测， percentage, classifier



ANN(Approximate Nearest Neighbor)

适用于无标记数据

相关程度的近似



Fixed radius nearest neighbor

在KNN算法基础上，限制距离



## 专业名词与缩写

- **SFM**(Structure From Motion)：
- **NVS**(Novel View Synthesis)
- **MVS**(Multi-view stereo)：

相机参数：

* $pose$：位姿，包括相机位置 position 以及相机朝向 orientation
* 



## 评价指标

- PSNR(Peak Signal-to-Noise Ratio)：峰值信噪比，值越高，图像质量越好
  - 高于40dB：质量极高
  - 30-40dB：质量较高
  - 20-30dB：质量较差
  - 低于20dB：质量不可接受
- SSIM (Structural SIMilarity) ：结构相似性，通过对比亮度、对比度和结构，衡量图片的相似程度，取值$[-1, 1]$，越高，两个图像越相似
- LPIPS：使用预训练的深度网络提取图像特征，计算特征间距离，评估感知相似度，越低相似度越高

