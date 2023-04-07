## KeyPose: Multi-View 3D Labeling and Keypoint Estimation for Transparent Objects

> 来源：2020年CVPR
>
> [ArXiv](https://arxiv.org/abs/1912.02805)
>
> [Github](https://github.com/google-research/google-research/tree/master/keypose)

#### Abstract

省流版：通过RGB相机建立了一种简单的方法来捕捉和标记桌面物体上的三维关键点，并提出了**KeyPose**的深度神经网络

> Estimating the 3D pose of desktop objects is crucial for applications such as robotic manipulation. Many existing approaches to this problem require a depth map of the object for both training and prediction, which restricts them to opaque, lambertian objects that produce good returns in an RGBD sensor . In this paper we forgo using a depth sensor in favor of raw stereo input. We address two problems: first, we establish an easy method for capturing and labeling 3D keypoints on desktop objects with an RGB camera; and second, we develop a deep neural network, called KeyPose, that learns to accurately predict object poses using 3D keypoints, from stereo input, and works even for transparent objects. To evaluate the performance of our method, we create a dataset of 15 clear objects in five classes, with 48K 3D-keypoint labeled images. We train both instance and category models, and show generalization to new textures, poses, and objects. KeyPose surpasses state-of-the-art performance in 3D pose estimation on this dataset by
> factors of 1.5 to 3.5, even in cases where the competing method is provided with ground-truth depth. Stereo input is essential for this performance as it improves results compared to using monocular input by a factor of 2. We will release a public version of the data capture and labeling pipeline, the transparent object database, and the KeyPose models and evaluation code.
>
> 估计物体的3D姿态对于机器人操作等应用程序至关重要。许多现有的解决该问题的方法都需要对象的**深度图**来进行训练和预测，这将它们限制在不透明的对象上，这些对象在RGBD传感器中产生良好的回报。在本文中，我们放弃使用深度传感器，以支持原始立体声输入。我们解决了两个问题:首先，我们通过RGB相机建立了一种简单的方法来捕捉和标记桌面物体上的三维关键点；其次，我们开发了一个名为**KeyPose**的深度神经网络，它可以从立体输入中学习使用3D关键点准确预测物体的姿势，甚至可以用于透明物体。

#### Introduction

省流版：对于透明物体，使用深度传感器无法获得准确的深度信息以及对象 CAD 模型，因此提出了一种从立体 RGB 图像中对（透明）3D 对象进行基于关键点的姿态估计的方法。立体 RGB 图像我的理解就是 不同位置的一组图像序列，后续介绍方法时会有图。本文的贡献就是**采集数据集的方法**、**数据集**、**KeyPose**。

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

![Data capturing pipeline](../../static/img/Data_capturing_pipeline.png)

#### 贡献1 采集数据集的方法

由于关键点深度的不确定性，很难或不可能在单个 RGB 图像中手动标记 3D 关键点。因此利用多视图几何将 2D 标签从少量图像提升为对象未移动的一组图像的 3D 标签。





参考文献

Clear-grasp: 3d shape estimation of transparent objects for manip-ulation.

Discovery of latent 3d keypoints via end-to-end geometric reasoning.