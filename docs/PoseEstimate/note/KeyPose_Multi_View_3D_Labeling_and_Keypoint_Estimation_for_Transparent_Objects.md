## KeyPose: Multi-View 3D Labeling and Keypoint Estimation for Transparent Objects

> 来源：2020年CVPR
>
> [ArXiv](https://arxiv.org/abs/1912.02805)
>
> [Github](https://github.com/google-research/google-research/tree/master/keypose)

![KeyPose_Multi_View_3D_Labeling_and_Keypoint_Estimation_for_Transparent_Objects](../../static/img/KeyPose_Multi_View_3D_Labeling_and_Keypoint_Estimation_for_Transparent_Objects/Title.png#pic_center)
#### Abstract

通过RGB相机建立了一种简单的方法来捕捉和标记桌面物体上的三维关键点，并提出了**KeyPose**的深度神经网络

> Estimating the 3D pose of desktop objects is crucial for applications such as robotic manipulation. Many existing approaches to this problem require a depth map of the object for both training and prediction, which restricts them to opaque, lambertian objects that produce good returns in an RGBD sensor . In this paper we forgo using a depth sensor in favor of raw stereo input. We address two problems: first, we establish an easy method for capturing and labeling 3D keypoints on desktop objects with an RGB camera; and second, we develop a deep neural network, called KeyPose, that learns to accurately predict object poses using 3D keypoints, from stereo input, and works even for transparent objects. To evaluate the performance of our method, we create a dataset of 15 clear objects in five classes, with 48K 3D-keypoint labeled images. We train both instance and category models, and show generalization to new textures, poses, and objects. KeyPose surpasses state-of-the-art performance in 3D pose estimation on this dataset by
> factors of 1.5 to 3.5, even in cases where the competing method is provided with ground-truth depth. Stereo input is essential for this performance as it improves results compared to using monocular input by a factor of 2. We will release a public version of the data capture and labeling pipeline, the transparent object database, and the KeyPose models and evaluation code.
>
> 估计物体的3D姿态对于机器人操作等应用程序至关重要。许多现有的解决该问题的方法都需要对象的**深度图**来进行训练和预测，这将它们限制在不透明的对象上，这些对象在RGBD传感器中产生良好的回报。在本文中，我们放弃使用深度传感器，以支持原始立体声输入。我们解决了两个问题:首先，我们通过RGB相机建立了一种简单的方法来捕捉和标记桌面物体上的三维关键点；其次，我们开发了一个名为**KeyPose**的深度神经网络，它可以从立体输入中学习使用3D关键点准确预测物体的姿势，甚至可以用于透明物体。

#### Introduction

对于透明物体，使用深度传感器无法获得准确的深度信息以及对象 CAD 模型，因此提出了一种从立体 RGB 图像中对（透明）3D 对象进行基于关键点的姿态估计的方法。立体 RGB 图像我的理解就是 不同位置的一组图像序列，后续介绍方法时会有图。本文的贡献就是**采集数据集的方法**、**数据集**、**KeyPose**。

>Estimating the position and orientation of 3D objects is one of the core problems in computer vision applications that involve object-level perception such as augmented reality (AR) and robotic manipulation. Rigid objects with a known model can be described by 4D pose, 6D pose, and 9D pose where scale is predicted. A more flexible method uses **3D keypoints**, which can handle articulated and deformable objects such as the human hand or body. While some of these methods predict 3D keypoints from a single RGB image, others use RGBD data collected by a depth sensor to achieve better accuracy. Unfortunately, existing commercial depth sensors, such as projected light or time-of-flight (ToF) sensors, assume that objects have opaque, lambertian surfaces that can support diffuse reflection from the sensor. Depth sensing fails when these conditions do not hold, e.g., for transparent or shiny metallic objects. 
>
>估计 3D 对象的位置和方向是计算机视觉应用程序的核心问题之一，涉及对象级感知，例如增强现实 (AR) 和机器人操作。具有已知模型的刚性对象可以通过 4D 姿态（例如，车辆）、6D 姿态和预测比例的 9D 姿态来描述。一种更灵活的方法使用 **3D 关键点（文章采用的方法）**，它可以处理铰接和可变形的物体，例如人的手或身体。虽然其中一些方法从单个 RGB 图像预测 3D 关键点，但其他方法使用深度传感器收集的 RGBD 数据来获得更高的准确性。不幸的是，现有的商业深度传感器，例如投射光或飞行时间 (ToF) 传感器，假设物体具有不透明的朗伯表面，可以支持来自传感器的漫反射。当这些条件不成立时，深度感应就会失败，例如，对于透明或闪亮的金属物体。
>
>In this paper, we present the first method of keypoint-based pose estimation for (transparent) 3D objects from stereo RGB images. There are several challenges: first, **there is no available large-scale dataset for transparent 3D object pose estimation** from stereo images with annotated keypoints. Datasets such as _NYUDepth v2_ lack an notations for precise pose of each individual objects, while other datasets such as *LabelFusion*, *YCB dataset* and *REAL275* annotate monocular RGBD images of opaque objects. The second challenge is **the annotation of pose of transparent 3D objects**. Existing datasets such as require accurate depth information as well as an object CAD model so that alignment algorithms such as *iterative closest point (ICP)* can be applied. The third challenge is how to leverage only RGB images for 3D keypoint estimation, thus obviating the need for a depth sensor.
>
>在本文中，我们提出了一种从立体 RGB 图像中对（透明）3D 对象进行基于关键点的姿态估计的方法。存在几个挑战：首先，**没有可用的大规模数据**集用于从带有注释关键点的立体图像进行透明 3D 对象姿态估计。 NYUDepth v2 等数据集缺乏对每个单独对象的精确姿势的注释，而其他数据集，如 LabelFusion、YCB 数据集 和 REAL275则对不透明对象的单目 RGBD 图像进行注释。第二个挑战是**透明 3D 物体姿态的标注**。现有数据集需要准确的深度信息以及对象 CAD 模型，以便可以[应用迭代最近点 (ICP)]() 等对齐算法。第三个挑战是**如何仅利用 RGB 图像进行 3D 关键点估计**，从而避免对深度传感器的需求。
>
>To address the challenges regarding data acquisition and annotation, we introduce an efficient method of capturing and labeling stereo RGB images for transparent (and other) objects. Although our method does not need them, we also capture and register depth maps of the object, for both the transparent object and its opaque twin, registered with the stereo images; we use a robotic arm to help automate this process. The registered opaque depth allows us to compare to methods that require depth maps as input such as *DenseFusion*. Following the proposed data capturing and labeling method, we constructed a large dataset consisting of 48k images from 15 transparent object instances. We call this dataset TOD (Transparent Object Dataset).
>
>为了解决有关数据采集和注释的挑战，我们引入了一种有效的方法来捕获和标记透明（和其他）对象的立体 RGB 图像。尽管我们的方法不需要它们，但我们还捕获并配准对象的深度图，对于透明对象及其不透明孪生对象，都与立体图像配准；我们使用机械臂来帮助自动化这个过程。注册的不透明深度允许我们与需要深度图作为输入的方法进行比较，例如 DenseFusion。按照提出的数据捕获和标记方法，我们构建了一个大型数据集，其中包含来自 15 个透明对象实例的 48k 图像。我们称此数据集为 *TOD（透明对象数据集）*。
>
>To reduce the requirement on reliable depth, we propose a deep model, **KeyPose**, that predicts 3D keypoints on transparent objects from cropped stereo RGB input. The crops are obtained from a detection stage that we assume can loosely bound objects. The model determines depth implicitly by combining information from the image pair, and predicting the 3D positions of keypoints for object instances and classes. After training on TOD, we compare KeyPose to the best existing RGB and RGBD methods and find that it vastly outperforms them on this dataset. In summary, we make the following contributions:
>
>1. A pipeline to label 3D keypoints on real-world objects, including transparent objects that does not require depth images, thus making learning-based 3D estimation of previously unknown objects possible without simulation data or accurate depth images. This pipeline supports a twin-opaque technique to enable comparison with models that require depth input.
>2. A dataset of 15 transparent objects in 6 classes, labeled with relevant 3D keypoints, and comprising 48k stereo and RGBD images with both transparent and opaque depth. This dataset can also be used in other transparent 3D object applications.
>3. A deep model, KeyPose, that predicts 3D keypoints on these objects with high accuracy using RGB stereo input only, and even outperforms methods which useground-truth depth input.

#### Related Work

##### 4D/6D/9D Pose Representation

现有的 4D/6D/9D 姿态估计技术通常可以根据 3D CAD 模型用于训练还是推理来分类。将观察到的 RGB 图像与渲染的 CAD 模型图像对齐，或者使用 ICP等算法将观察到的 3D 点云与 3D CAD 模型点云对齐，或者从 3D 渲染混合现实数据CAD 模型作为额外的训练数据。



#### Transparent Object Dataset (TOD)
##### 1.Data Collection with a Robot
由于关键点深度的不确定性，很难或不可能在单个 RGB 图像中手动标记 3D 关键点。因此利用多视图几何将 2D 标签从少量图像提升为对象未移动的一组图像的 3D 标签。

![Data_capturing_pipeline](../../static/img/KeyPose_Multi_View_3D_Labeling_and_Keypoint_Estimation_for_Transparent_Objects/Data_capturing_pipeline.png#pic_center)
使用具有已知参数的立体相机按顺序捕获图像，用机械臂移动相机（也可以用手移动它）。为了估计相机相对于世界的姿态，我们使用AprilTags（就是对应图中的虚线部分，可以理解成从另外一个角度观察整个过程）建立了一个平面形式，可以在图像中识别它，并从它们的已知位置估计相机姿态。

从图像序列中的每一个小部分，我们在对象上标记 2D 关键点。多视图几何的优化给出了关键点的 3D 位置，可以将其重新投影到序列中的所有图像。为了增加多样性，在对象下方放置了各种纹理
##### 2.Keypoint Labeling and Automatic Propagation
为了准确地构建这个数据集，我们需要解决不同的错误来源。首先，由于 AprilTag 检测在寻找标签位置方面并不完美，我们将这些标签散布在目标上以生成用于相机姿势估计的大基线。其次，由于人类在 2D 图像上标记关键点会引入错误，因此我们在相机姿势上使用最远点算法以确保用于从 2D 到 3D 的注释图像具有较大的基线。虽然 3D 关键点的绝对真实值是未知的，但我们可以估计标记误差，给定 AprilTags 和 2D 注释的已知重投影误差。使用基于重投影误差的蒙特卡罗模拟，我们计算出标记的 3D 关键点的随机误差约为3.4mmRMSE。(看不懂)

#### Predicting 3D Keypoints from RGB Stereo (KeyPose)
##### 1.Data Input to the Training Process
![Data Input to the Training Process](../../static/img/KeyPose_Multi_View_3D_Labeling_and_Keypoint_Estimation_for_Transparent_Objects/Data_Input_to_the_Training_Process.png#pic_center)
假设检测阶段大致确定了物体的位置。从这个边界框，我们从左侧图像裁剪一个固定大小的矩形，并从右侧图像裁剪一个相同高度的相应矩形，保留对极几何形状。
由于右边的物体图像向左边偏移48到96像素，为了限制矩形扩展，我们将右侧裁剪水平偏移 30像素，将视差更改为18-66像素。每次裁剪的输入尺寸为180×120像素。
模型处理输入图像，为每个关键点生成关键点的 UV (2D) 图像位置和对深度进行编码的视差D，它是左右关键点的偏移量（以像素为单位）。UVD 三元组通过以下方式对 3D XYZ 坐标进行编码：Q := UVD 7→ XYZ，其中 Q 是由相机参数确定的重投影矩阵。我们使用这些 XYZ 位置作为标签来生成训练错误，通过投影回相机图像并比较 UVD 差异。重投影像素误差是一种稳定的、物理上可实现的误差方法，广泛用于多视图几何（见参考文献3）。直接比较 3D 误差会引入很大的偏差，因为它们随距离呈二次方增长，压倒了较近物体的误差。
为了鼓励泛化，我们对输入图像进行几何和光度增强。
##### 2.Architecture for 3D Pose Estimation
**Stereo for Implicit Depth**: Use stereo images to introduce depth information to the model.通过立体图像隐式引入深度信息
**Early Fusion**: Combine information from the two image crops as early as possible. Let the deep neural network determine disparity implicitly, rather than forming explicit correlations.尽可能早地结合两张裁剪的图像，通过网络隐式地确定视差 (做了对比实验)
**Broad Context**: Extend the spatial context of each key-point as broadly as possible, to take advantage of any relevant shape information of the object.尽可能广泛地扩展每个关键点的空间上下文
![Data Input to the Training Process](../../static/img/KeyPose_Multi_View_3D_Labeling_and_Keypoint_Estimation_for_Transparent_Objects/Early_fusion_architecture.png#pic_center)
立体图像被堆叠并送入一组指数扩张的 3x3 卷积([参考文献4：通过扩张卷积进行多尺度上下文聚合]())，这些卷积扩展了预测关键点的上下文，同时保持分辨率不变。两个这样的分组确保每个关键点的上下文完全混合。
我们研究了两种投影方法：1.直接回归。三个 1x1 卷积层产生 N × 3 数字 UVD 坐标，其中 N 是关键点的数量。2.热图。对于每个关键点i，CNN 层生成热图，然后是空间 softmax 生成概率图 probi，然后集成以获得 UV 坐标。计算视差热图，与概率图进行卷积，并集成以产生视差。
![Data Input to the Training Process](../../static/img/KeyPose_Multi_View_3D_Labeling_and_Keypoint_Estimation_for_Transparent_Objects/Late_fusion_architecture.png#pic_center)
为了测试早期融合的效果，我们还实现了后期融合模型，其中孪生扩张 CNN 块分别预测左右图像的 UV 关键点。然后使用标准立体几何生成 3D 关键点预测。
##### 3. Losses(损失函数)
使用了三种损失：直接关键点 UVD 损失、投影损失和局部损失。还排列了总损失并对对称关键点取最小值
**Keypoint loss**
预测的 (UVD) 和标记的 (UVD ∗) 像素值通过平方误差进行比较 $Lkp = X i∈kps kUVD i − UVD ∗ i k2$。
直接使用（XYZ）3D损失会导致误差随距离呈二次方增长，会给模型性能带来了很大的偏差。
**Projection loss**
预测的 UVD 值被转换为 3D 点，然后重新投影到用于创建 3D 点的广泛分离的视图。预测的和标记的 UV 重新投影之间的差异是损失的平方。设 Pj 为投影函数，Q := UVD 7→ XYZ 。然后 Lproj = X i∈keypts X j∈views kPjQ(UVD i) − PjQ(UVD ∗i )k2
**Locality loss**
虽然关键点的位置是由热图概率图估计出来的，但该图可能不是单模态的，可能会有远离真正关键点位置的高概率。这种损失促使概率图在关键点周围进行定位。
N是以关键点i的标记UV∗i坐标为中心的圆形正态分布，其标准差为σ。N是一个归一化的倒数1-N / max(N )。(4) 当预测的UV概率集中在UV标签附近时，这个损失给出了一个非常低的值。我们使用10个像素的σ

总的损失被定义为加权和Ltotal = Lkp + αLproj + 0.001Lloc (5) Lloc上的小权重将概率分布推向正确的局部形式，同时允许在必要时有散开的空间。为了稳定起见，对Lproj适用一个课程是很重要的。权重α在训练步骤的[1/3, 2/3]区间内从0升至2.5，以使预测的UVD值稳定下来。否则，收敛会很困难，因为重投误差梯度最初可能非常大。

**Permutation for symmetric keypoints**
对称对象可能会导致关键点 ID 之间出现混叠。例如，图 9 中的树对象在绕其垂直轴旋转 180° 时无法区分。因此，从姿势估计器的角度来看，放置在对象上的关键点可能会获得不同的、无法区分的位置。

我们通过允许在损失函数中排列相关关键点 ID 来处理关键点对称性。例如，在树的情况下，关键点 ID 有两个允许的排列，[1, 2, 3, 4] 和 [1, 2, 4, 3]。对这些排列中的每一个都评估 Ltotal，并选择最小值作为最终损失

#### 参考文献

Clear-grasp: 3d shape estimation of transparent objects for manip-ulation.

Discovery of latent 3d keypoints via end-to-end geometric reasoning.

3. Geometric loss functions for camera pose regression with deep learning.
4. Multi-scale context aggregation by dilated convolutions.