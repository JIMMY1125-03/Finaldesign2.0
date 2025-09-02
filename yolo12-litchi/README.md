# 【大作业-48】基于深度学习的荔枝病虫害检测

🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳

视频地址：[手把手教你使用YOLO11实现车辆检测与追踪系统_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1nzzdYwE2g/)

🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳

大家好！我们从今天开始偏向一下农业方向的，国家也提出提升农业新质生产力 以科技赋能备春耕。所以本次我们带来的自动化程序是基于yolo12和yolo11的荔枝病虫害检测，本次带来的检测的类别包含下面的种类。

* **Anthrax** - 炭疽病
* **Felt_disease** - 毡病
* **Phytophthora_downy** - 致病疫霉霜霉病（“Phytophthora”是致病疫霉属，常引起植物霜霉病）
* **Lychee_Tsubaki_elephant** - 荔枝山茶象
* **Pedicle_borer** - 花柄螟
* **Turtle_backed_beetle** - 龟背甲虫

以下是部分数据示例。

![train_batch0](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/202503271501183.jpg)

下面是部分实现效果，支持视频和图像检测。

![image-20250327151442223](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/202503271514710.png)

![image-20250327151502687](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/202503271515749.png)

## 项目实战

进行项目实战之前请务必安装好pytorch和miniconda。

不会的小伙伴请看这里：[Python项目配置前的准备工作-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/144233262?sharetype=blogdetail&sharerId=144233262&sharerefer=PC&sharesource=ECHOSON&spm=1011.2480.3001.8118)

<font color='red'>配置之前首先需要下载项目资源包，项目资源包请看从上方视频的置顶评论中或者是博客绑定资源获取即可。</font>

![image-20250111195350376](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250111195350376.png)

### 环境配置

环境配置请看这里：[【肆十二】YOLO系列代码环境配置统一流程-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/145405669)

### 本地模型训练

模型训练使用的脚本为` step1_start_train.py `，进行模型训练之前，请先按照配置好你本地的数据集。数据集在` ultralytics\cfg\datasets\A_my_data.yaml`目录下，你需要将数据集的根目录更换为你自己本地的目录。

![image-20241204100852481](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204100852481.png)

![image-20250109222911440](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250109222911440.png)

更换之后修改训练脚本配置文件的路径，直接右键即可开始训练。

![image-20250109223259429](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250109223259429.png)

训练开始前如果出现报错，有很大的可能是数据集的路径没有配置正确，请检查数据集的路径，保证数据集配置没有问题。训练之后的结果将会保存在runs目录下。

![image-20241204101214326](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204101214326.png)

### GPU服务器训练（可选）

目前蓝耘GPU可以薅羊毛，推荐小伙伴从这个网站使用GPU云来进行训练，新用户注册会获得30元的代金券。

注册地址：[蓝耘GPU智算云平台](https://cloud.lanyun.net/#/registerPage?promoterCode=0118 )

服务器使用指南：[手把手教你使用服务器训练AI模型_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1TuxLeVED6?vd_source=2f9a4e63109c3db3be5e8078e5111776&spm_id_from=333.788.videopod.sections)

### 模型测试

模型的测试主要是对map、p、r等指标进行计算，使用的脚本为` step2_start_val.py`，模型在训练的最后一轮已经执行了测试，其实这个步骤完全可以跳过，但是有的朋友可能想要单独验证，那你只需要更改测试脚本中的权重为你自己所训练的权重路径，即可单独进行测试。

![image-20241204101429118](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204101429118.png)

### 图形化界面封装

图形化界面进行了升级，本次图形化界面的开发我们使用pyside6来进行开发。**PySide6** 是一个开源的Python库，它是Qt 6框架的Python绑定。Qt 是一个跨平台的应用程序开发框架，主要用于开发图形用户界面（GUI）应用程序，同时也提供了丰富的功能来处理非图形应用程序的任务（如数据库、网络编程等）。PySide6 使得开发者能够使用 Python 编写 Qt 6 应用程序，因此，它提供了Python的灵活性和Qt 6的强大功能。图形化界面提供了图片和视频检测等多个功能，图形化界面的程序为` step3_start_window_track.py `。

如果你重新训练了模型，需要替换为你自己的模型，请在这里进行操作。

![image-20241204101842858](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204101842858.png)

如果你想要对图形化界面的题目、logo等进行修改，直接在这里修改全局变量即可。

![image-20241204101949741](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204101949741.png)

登录之后上传图像或者是上传视频进行检测即可。

![](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/202503271515469.png)

![image-20241211204753525](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241211204753525.png)

对于web界面的封装，对应的python文件是`web_demo.py`，我们主要使用gradio来进行开发，gradio，详细的代码如下：

```python
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：step3_start_window_track.py 
@File    ：web_demo.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：TODO 添加文件描述
@Date    ：2024/12/11 20:25 
'''
import gradio as gr
import PIL.Image as Image

from ultralytics import ASSETS, YOLO

model = YOLO("runs/yolo11s/weights/best.pt")


def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLO11 model with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="基于YOLO11的垃圾检测系统",
    description="Upload images for inference.",
    # examples=[
    #     [ASSETS / "bus.jpg", 0.25, 0.45],
    #     [ASSETS / "zidane.jpg", 0.25, 0.45],
    # ],
)

if __name__ == "__main__":
    # iface.launch(share=True)
    # iface.launch(share=True)
    iface.launch()
```

## 文档

### 背景与意义

荔枝作为我国华南地区最重要的经济林果树之一，种植面积和产量均居世界首位。然而，荔枝在种植过程中容易受到多种病虫害的侵袭，常见的病虫害包括荔枝炭疽病、荔枝毛毡病、荔枝叶瘿蚊、荔枝藻斑病和荔枝煤烟病等。这些病虫害不仅种类繁多，而且发病周期长、防治困难，严重影响荔枝的产量和品质。传统的病虫害检测方法主要依靠人工根据经验现场观察判定，存在判定效率低、成本高、受主观性影响等问题，无法满足现代农业大规模、实时检测的需求。

近年来，随着深度学习技术的快速发展，基于图像识别的目标检测技术在农业领域得到了广泛应用。YOLO（You Only Look Once）系列模型因其检测速度快、准确率高，特别适合应用于移动端和嵌入式设备。然而，传统的YOLO模型在复杂背景下表现不佳，尤其是在自然环境下，背景的多样性和非结构化特征会对检测精度产生影响。此外，荔枝病虫害的范围大小不一，也会影响检测的准确率。

使用YOLOv12模型进行荔枝病虫害检测，可以实现快速、准确地识别病虫害类型，大大提高了检测效率。与传统的人工检测方法相比，基于深度学习的模型能够实时处理大量图像数据，减少人工干预，提高检测的准确性和客观性。

改进后的YOLOv12模型在多场景下表现出良好的适应性和鲁棒性。无论是自然环境还是实验室环境，该模型都能有效检测荔枝病虫害，为实际应用提供了可靠的技术支持。

### 相关文献综述

随着深度学习技术的快速发展，基于卷积神经网络（CNN）的图像识别技术在农业领域得到了广泛应用，尤其是在农作物病虫害检测方面。近年来，YOLO（You Only Look Once）系列模型因其检测速度快、准确率高，逐渐成为农业病虫害检测领域的热门选择。
在荔枝病虫害检测方面，已有研究利用改进的YOLO模型取得了显著成果。例如，有研究提出了一种基于改进YOLO v4的荔枝病虫害检测模型，该模型使用GhostNet作为主干网络，并结合CBAM注意力机制，有效提高了检测精度，同时保持了模型的轻量化。此外，另一项研究针对荔枝虫害小目标检测问题，提出了一种基于改进YOLOv10n的轻量化模型YOLO-LP，通过优化主干网络和颈部网络结构，以及引入新的损失函数，显著提升了小目标检测的准确率。
除了模型改进，数据集的构建和数据增强方法也对检测效果起到了重要作用。例如，在构建荔枝病虫害图像数据集时，研究人员通过在不同光照、天气条件下采集图像，并采用多种数据增强技术，提高了模型对复杂自然环境的适应性。此外，基于深度学习的荔枝虫害识别技术已在实际场景中得到应用，证明了其在真实复杂环境中进行有效识别的可行性。
综上所述，YOLO系列模型在荔枝病虫害检测领域展现出了良好的应用前景。未来的研究可以进一步探索模型的优化和改进，以应对更加复杂的自然环境和多样化的病虫害类型。同时，结合其他深度学习技术和数据增强方法，有望进一步提高检测的准确率和效率。

### 本文算法介绍

#### yolo12算法介绍

![image-20250312171139485](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250312171139485.png)

yolo12算法在原先yolo11的基础上进行了微调，引入了一种以注意力为中心的架构，它不同于以往YOLO 模型中使用的基于 CNN 的传统方法，但仍保持了许多应用所必需的实时推理速度。该模型通过对注意力机制和整体网络架构进行新颖的方法创新，实现了最先进的物体检测精度，同时保持了实时性能。

- **区域注意机制**：一种新的自我注意方法，能有效处理大的感受野。它可将[特征图](https://www.ultralytics.com/glossary/feature-maps)横向或纵向划分为*l 个*大小相等的区域（默认为 4 个），从而避免复杂的操作，并保持较大的有效感受野。与标准自注意相比，这大大降低了计算成本。
- **剩余效率层聚合网络（R-ELAN）:**基于 ELAN 的改进型特征聚合模块，旨在解决优化难题，尤其是在以注意力为中心的大规模模型中。R-ELAN 引入了
  - 具有缩放功能的块级残差连接（类似于图层缩放）。
  - 重新设计的特征聚合方法可创建类似瓶颈的结构。
- **优化注意力架构:** YOLO12 简化了标准关注机制，以提高效率并与YOLO 框架兼容。这包括
  - 使用 FlashAttention 尽量减少内存访问开销。
  - 去除位置编码，使模型更简洁、更快速。
  - 调整 MLP 比例（从通常的 4 调整为 1.2 或 2），以更好地平衡注意力层和前馈层之间的计算。
  - 减少堆叠区块的深度，提高优化效果。
  - 酌情利用卷积运算，提高计算效率。
  - 在注意力机制中加入 7x7 可分离卷积（"位置感知器"），对位置信息进行隐式编码。

在配置文件上，他和yolo11的区别就比较小了，主要体现在下图红框中所示的区域。

![image-20250312170003994](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250312170003994.png)

其中A2C2F就是yolo12中所提出的主要模块，A表示的含义是Area Attention。为了克服传统自注意力机制计算复杂度高的问题，YOLOv12通过创新的区域注意力模块（Area Attention，A2），分辨率为(H, W)的特征图被划分为l个大小为(H/l, W)或(H, W/l)的段。这消除了显式的窗口划分，仅需要简单的重塑操作，从而实现更快的速度。将l的默认值设置为4，将感受野减小到原来的1/4，但仍保持较大的感受野。采用这种方法，注意力机制的计算成本从2n²hd降低到1/2n²hd。尽管存在n²的复杂度，但当n固定为640时（如果输入分辨率增加，则n会增加），这仍然足够高效，可以满足YOLO系统的实时要求。A2降低了注意力机制的计算成本，同时保持较大的感受野，显著提升了检测精度。如下图所示，右侧的区域所表示的就是作者提出的注意力机制，相当于是以较少的计算量就捕捉到了相关的区域。

![image-20250312170135607](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250312170135607.png)

下面是Area Attention的实现，如下。

```python
class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        """
        Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y
```

#### yolo11算法介绍

yolo系列已经在业界可谓是家喻户晓了，下面是yolo11放出的性能测试图，其中这种图的横轴为模型的速度，一般情况下模型的速度是通过调整卷积的深度和宽度来进行修改的，纵轴则表示模型的精度，可以看到在同样的速度下，11表现出更高的精度。

![image-20241024170914031](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241024170914031.png)

YOLO架构的核心由三个基本组件组成。首先，主干作为主要特征提取器，利用卷积神经网络将原始图像数据转换成多尺度特征图。其次，颈部组件作为中间处理阶段，使用专门的层来聚合和增强不同尺度的特征表示。第三，头部分量作为预测机制，根据精细化的特征映射生成目标定位和分类的最终输出。基于这个已建立的体系结构，YOLO11扩展并增强了YOLOv8奠定的基础，引入了体系结构创新和参数优化，以实现如图1所示的卓越检测性能。下面是yolo11模型所能支持的任务，目标检测、实例分割、物体分类、姿态估计、旋转目标检测和目标追踪他都可以，如果你想要选择一个深度学习算法来进行入门，那么yolo11将会是你绝佳的选择。

![image-20241024171109729](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241024171109729.png)

为了能够让大家对yolo11网络有比较清晰的理解，下面我将会对yolo11的结构进行拆解。

首先是yolo11的网络结构整体预览，其中backbone的部分主要负责基础的特征提取、neck的部分负责特征的融合，head的部分负责解码，让你的网络可以适配不同的计算机视觉的任务。

![image-20241024173654996](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241024173654996.png)

* 主干网络（BackBone）

  * Conv

    卷积模块是一个常规的卷积模块，在yolo中使用的非常多，可以设计卷积的大小和步长，代码的详细实现如下：

    ```python
    class Conv(nn.Module):
        """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    
        default_act = nn.SiLU()  # default activation
    
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            """Initialize Conv layer with given arguments including activation."""
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
        def forward(self, x):
            """Apply convolution, batch normalization and activation to input tensor."""
            return self.act(self.bn(self.conv(x)))
    
        def forward_fuse(self, x):
            """Perform transposed convolution of 2D data."""
            return self.act(self.conv(x))
    ```

  * C3k2

    C3k2块被放置在头部的几个通道中，用于处理不同深度的多尺度特征。他的优势有两个方面。一个方面是这个模块提供了更快的处理:与单个大卷积相比，使用两个较小的卷积可以减少计算开销，从而更快地提取特征。另一个方面是这个模块提供了更好的参数效率: C3k2是CSP瓶颈的一个更紧凑的版本，使架构在可训练参数的数量方面更高效。

    C3k2模块主要是为了增加特征的多样性，其中这块模块是由C3k模块演变而来。它通过允许自定义内核大小提供了增强的灵活性。C3k的适应性对于从图像中提取更详细的特征特别有用，有助于提高检测精度。C3k的实现如下。

    ```python
    class C3k(C3):
        """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
    
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
            """Initializes the C3k module with specified channels, number of layers, and configurations."""
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
            self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
    ```

    如果将c3k中的n设置为2，则此时的模块即为C3K2模块，网络结构图如下所示。

    ![image-20241025121912923](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241025121912923.png)

    该网络的实现代码如下。

    ```python
    class C3k2(C2f):
        """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    
        def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
            """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
            super().__init__(c1, c2, n, shortcut, g, e)
            self.m = nn.ModuleList(
                C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
            )
    ```

  * C2PSA

    PSA的模块起初在YOLOv10中提出，通过自注意力的机制增加特征的表达能力，相对于传统的自注意力机制而言，计算量又相对较小。网络的结构图如下所示，其中图中的mhsa表示的是多头自注意力机制，FFN表示前馈神经网络。

    ![image-20241025122617233](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241025122617233.png)

    

  在这个基础上添加给原先的C2模块上添加一个PSA的旁路则构成了C2PSA的模块，该模块的示意图如下。

  ![image-20241025122752167](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241025122752167.png)

  网络实现如下：

  ```python
  class C2PSA(nn.Module):
      """
      C2PSA module with attention mechanism for enhanced feature extraction and processing.
  
      This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
      capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.
  
      Attributes:
          c (int): Number of hidden channels.
          cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
          cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
          m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.
  
      Methods:
          forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.
  
      Notes:
          This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
  
      Examples:
          >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
          >>> input_tensor = torch.randn(1, 256, 64, 64)
          >>> output_tensor = c2psa(input_tensor)
      """
  
      def __init__(self, c1, c2, n=1, e=0.5):
          """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
          super().__init__()
          assert c1 == c2
          self.c = int(c1 * e)
          self.cv1 = Conv(c1, 2 * self.c, 1, 1)
          self.cv2 = Conv(2 * self.c, c1, 1)
  
          self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
  
      def forward(self, x):
          """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
          a, b = self.cv1(x).split((self.c, self.c), dim=1)
          b = self.m(b)
          return self.cv2(torch.cat((a, b), 1))
  
  ```

* 颈部网络（Neck）

  * upsample

    这里是一个常用的上采样的方式，在YOLO11的模型中，这里一般使用最近邻差值的方式来进行实现。在 `torch`（PyTorch）中，`upsample` 操作是用于对张量（通常是图像或特征图）进行**上采样**（增大尺寸）的操作。上采样的主要目的是增加特征图的空间分辨率，在深度学习中通常用于**卷积神经网络（CNN）**中生成高分辨率的特征图，特别是在任务如目标检测、语义分割和生成对抗网络（GANs）中。

    PyTorch 中的 `torch.nn.functional.upsample` 在较早版本提供了上采样功能，但在新的版本中推荐使用 `torch.nn.functional.interpolate`，功能相同，但更加灵活和标准化。

    主要参数如下：

    `torch.nn.functional.interpolate` 函数用于上采样，支持不同的插值方法，常用的参数如下：

    ```python
    torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
    ```

    - `input`：输入的张量，通常是 4D 的张量，形状为 `(batch_size, channels, height, width)`。

    - `size`：输出的目标尺寸，可以是整型的高度和宽度（如 `(height, width)`），表示希望将特征图调整到的具体尺寸。

    - `scale_factor`：上采样的缩放因子。例如，`scale_factor=2` 表示特征图的高度和宽度都扩大 2 倍。如果设置了 `scale_factor`，则不需要再设置 `size`。

    - ```
      mode
      ```

      ：插值的方式，有多种可选插值算法：

      - `'nearest'`：最近邻插值（默认）。直接复制最近的像素值，计算简单，速度快，但生成图像可能比较粗糙。
      - `'linear'`：双线性插值，适用于 3D 输入（即 1D 特征图）。
      - `'bilinear'`：双线性插值，适用于 4D 输入（即 2D 特征图）。
      - `'trilinear'`：三线性插值，适用于 5D 输入（即 3D 特征图）。
      - `'bicubic'`：双三次插值，计算更复杂，但生成的图像更平滑。

    - `align_corners`：在使用双线性、三线性等插值时决定是否对齐角点。如果为 `True`，输入和输出特征图的角点会对齐，通常会使插值结果更加精确。

  * Concat

    在YOLO（You Only Look Once）目标检测网络中，`concat`（连接）操作是用于将来自不同层的特征图拼接起来的操作。其作用是融合不同尺度的特征信息，以便网络能够在多个尺度上更好地进行目标检测。调整好尺寸后，沿着**通道维度**将特征图进行拼接。假设我们有两个特征图，分别具有形状 (H, W, C1) 和 (H, W, C2)，拼接后得到的特征图形状将是 (H, W, C1+C2)，即通道数增加了。一般情况下，在进行concat操作之后会再进行一次卷积的操作，通过卷积的操作可以将通道数调整到理想的大小。该操作的实现如下。

    ```python
    class Concat(nn.Module):
        """Concatenate a list of tensors along dimension."""
    
        def __init__(self, dimension=1):
            """Concatenates a list of tensors along a specified dimension."""
            super().__init__()
            self.d = dimension
    
        def forward(self, x):
            """Forward pass for the YOLOv8 mask Proto module."""
            return torch.cat(x, self.d)
    ```

* 头部（Head）

  YOLOv11的Head负责生成目标检测和分类方面的最终预测。它处理从颈部传递的特征映射，最终输出图像内对象的边界框和类标签。一般负责将特征进行映射到你对应的任务上，如果是检测任务，对应的就是4个边界框的值以及1个置信度的值和一个物体类别的值。如下所示。

  ```python
  # Ultralytics YOLO 🚀, AGPL-3.0 license
  """Model head modules."""
  
  import copy
  import math
  
  import torch
  import torch.nn as nn
  from torch.nn.init import constant_, xavier_uniform_
  
  from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
  
  from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
  from .conv import Conv, DWConv
  from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
  from .utils import bias_init_with_prob, linear_init
  
  __all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect"
  
  
  ```

基于上面的设计，yolo11衍生出了多种变种，如下表所示。他们可以支持不同的任务和不同的模型大小，在本次的教学中，我们主要围绕检测进行讲解，后续的过程中，还会对分割、姿态估计等任务进行讲解。

![image-20241024173356022](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241024173356022.png)

YOLOv11代表了CV领域的重大进步，提供了增强性能和多功能性的引人注目的组合。YOLO架构的最新迭代在精度和处理速度方面有了显著的改进，同时减少了所需参数的数量。这样的优化使得YOLOv11特别适合广泛的应用程序，从边缘计算到基于云的分析。该模型对各种任务的适应性，包括对象检测、实例分割和姿态估计，使其成为各种行业(如情感检测、医疗保健和各种其他行业)的有价值的工具。它的无缝集成能力和提高的效率使其成为寻求实施或升级其CV系统的企业的一个有吸引力的选择。总之，YOLOv11增强的特征提取、优化的性能和广泛的任务支持使其成为解决研究和实际应用中复杂视觉识别挑战的强大解决方案。

### 实验结果分析

#### 数据集介绍

本次我们我们使用的数据集为荔枝病虫害数据集，数据集的图像数量为300, 下面是数据集的分布。

![labels](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/202503271507311.jpg)

我在这里已经将数据按照yolo分割数据集格式进行了处理，大家只需要在配置文件种对本地的数据地址进行配置即可，如下所示。

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: G:/Upppppdate/0000-48/litchi_yolo_format/ # dataset root dir
train: images/train # train images (relative to 'path') 1052 images
val: images/test # val images (relative to 'path') 225 images
test: images/test # test images (relative to 'path') 227 images
# Classes
names: ['Anthrax', 'Felt_disease', 'Phytophthora_downy', 'Lychee_Tsubaki_elephant', 'Pedicle_borer', 'Turtle_backed_beetle']
```

下面是数据集的部分示例。

![train_batch1171](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/202503271506855.jpg)

![train_batch1170](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/202503271506491.jpg)

#### 实验结果分析

实验结果的指标图均保存在runs目录下， 大家只需要对实验过程和指标图的结果进行解析即可。

如果只指标图的定义不清晰，请看这个位置：[YOLO11模型指标解读-mAP、Precision、Recall_yolo11模型训练特征图-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/144097341)

![results-crack](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/results-crack.png)

train/box_loss（训练集的边界框损失）：随着训练轮次的增加，边界框损失逐渐降低，表明模型在学习更准确地定位目标。
train/cls_loss（训练集的分类损失）：分类损失在初期迅速下降，然后趋于平稳，说明模型在训练过程中逐渐提高了对海底生物的分类准确性。
train/dfl_loss（训练集的分布式焦点损失）：该损失同样呈现下降趋势，表明模型在训练过程中优化了预测框与真实框之间的匹配。
metrics/precision(B)（精确度）：精确度随着训练轮次的增加而提高，说明模型在减少误报方面表现越来越好。
metrics/recall(B)（召回率）：召回率也在逐渐上升，表明模型能够识别出更多的真实海底生物。
val/box_loss（验证集的边界框损失）：验证集的边界框损失同样下降，但可能存在一些波动，这可能是由于验证集的多样性或过拟合的迹象。
val/cls_loss（验证集的分类损失）：验证集的分类损失下降趋势与训练集相似，但可能在某些点上出现波动。
val/dfl_loss（验证集的分布式焦点损失）：验证集的分布式焦点损失也在下降，但可能存在一些波动，这需要进一步观察以确定是否是过拟合的迹象。
metrics/mAP50(B)（在IoU阈值为0.5时的平均精度）：mAP50随着训练轮次的增加而提高，表明模型在检测任务上的整体性能在提升。
metrics/mAP50-95(B)（在IoU阈值从0.5到0.95的平均精度）：mAP50-95的提高表明模型在不同IoU阈值下的性能都在提升，这是一个更严格的性能指标。

![PR_curve](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/202503271506707.png)

当iou阈值为0.5的时候，模型在测试集上的map可以达到右上角所示的数值。下面是一个预测图像，可以看出，我们的模型可以有效的预测出这些尺度比较多变的目标。

![val_batch0_pred](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/202503271506311.jpg)

![val_batch1_pred](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/202503271506709.jpg)

### 结论

本研究利用YOLOv12模型对常见的荔枝病虫害进行检测，取得了显著的研究成果。荔枝作为我国重要的经济作物，其病虫害的精准检测对于保障产量和品质具有重要意义。传统的人工检测方法存在效率低、成本高、主观性强等问题，难以满足现代农业的需求。近年来，深度学习技术尤其是卷积神经网络（CNN）在图像识别领域取得了巨大突破，为荔枝病虫害的自动化检测提供了新的解决方案。

在众多深度学习模型中，YOLO系列因其检测速度快、准确率高而备受关注。本研究基于YOLOv12模型，针对荔枝病虫害图像的特点进行了优化。通过构建高质量的数据集并采用数据增强技术，模型能够更好地适应复杂自然环境下的图像特征。此外，引入注意力机制和轻量化网络结构，进一步提高了模型的检测精度和运行效率。

实验结果表明，YOLOv12模型在荔枝病虫害检测中表现出了较高的准确率和鲁棒性。与传统方法相比，该模型能够快速、准确地识别多种病虫害类型，为实时监测和精准防治提供了有力支持。同时，模型的轻量化设计使其适合在移动终端和嵌入式设备上部署，具有广泛的应用前景。

未来的研究可以进一步优化模型结构，提高对小目标和复杂背景的检测能力。此外，结合多源数据和多模型融合技术，有望进一步提升检测的准确率和可靠性。总之，基于YOLOv12的荔枝病虫害检测模型为农业病虫害的智能化监测提供了新的思路和方法，具有重要的理论和实践价值。

### 参考文献

[1]  Zhang Y , Li H , Bu R ,et al.Fuzzy Multi-objective Requirements for NRP Based on Particle Swarm Optimization[C]//2020.DOI:10.1007/978-3-030-57881-7_13.

[2]  Zhao N , Cao M , Song C ,et al.Trusted Component Decomposition Based on OR-Transition Colored Petri Net[C]//International Conference on Artificial Intelligence and Security.Springer, Cham, 2019.DOI:10.1007/978-3-030-24268-8_41.
DOI: 10.1109/ACCESS.2020.2973568

[3] Song C, Chang H. RST R-CNN: a triplet matching few-shot remote sensing object detection framework[C]//Fourth International Conference on Computer Vision, Application, and Algorithm (CVAA 2024). SPIE, 2025, 13486: 553-568.

[4]   Zhou Q , Yu C . Point RCNN: An Angle-Free Framework for Rotated Object Detection[J]. Remote Sensing, 2022, 14.

[5]  Zhang, Y., Li, H., Bu, R., Song, C., Li, T., Kang, Y., & Chen, T. (2020). Fuzzy Multi-objective Requirements for NRP Based on Particle Swarm Optimization. *International Conference on Adaptive and Intelligent Systems*.

[6]   Li X , Deng J , Fang Y . Few-Shot Object Detection on Remote Sensing Images[J]. IEEE Transactions on Geoscience and Remote Sensing, 2021(99).

[7]   Su W, Zhu X, Tao C, et al. Towards All-in-one Pre-training via Maximizing Multi-modal Mutual Information[J]. arXiv preprint arXiv:2211.09807, 2022.

[8]   Chen Q, Wang J, Han C, et al. Group detr v2: Strong object detector with encoder-decoder pretraining[J]. arXiv preprint arXiv:2211.03594, 2022.

[9]   Liu, Shilong, et al. "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection." arXiv preprint arXiv:2303.05499 (2023).

[10] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

[11] Redmon J, Farhadi A. YOLO9000: better, faster, stronger[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 7263-7271.

[12] Redmon J, Farhadi A. Yolov3: An incremental improvement[J]. arXiv preprint arXiv:1804.02767, 2018.

[13] Tian Z, Shen C, Chen H, et al. Fcos: Fully convolutional one-stage object detection[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2019: 9627-9636.

[14] Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous separable convolution for semantic image segmentation[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 801-818.

[15] Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector[C]//Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14. Springer International Publishing, 2016: 21-37.

[16] Lin T Y, Dollár P, Girshick R, et al. Feature pyramid networks for object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 2117-2125.

[17] Cai Z, Vasconcelos N. Cascade r-cnn: Delving into high quality object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 6154-6162.

[18] Ren S, He K, Girshick R, et al. Faster r-cnn: Towards real-time object detection with region proposal networks[J]. Advances in neural information processing systems, 2015, 28.

[19] Wang R, Shivanna R, Cheng D, et al. Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems[C]//Proceedings of the web conference 2021. 2021: 1785-1797.

[20] Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.

---------------------------------------------------------------------------------------------------------------------

### 模型改进的基本流程（选看）

首先我们说说如何在yolo的基础模型上进行改进。

1. 在`block.py`或者`conv.py`中添加你要修改的模块，比如我在这里添加了se的类，包含了输入和输出的通道数。

   ![image-20250108112113879](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250108112113879.png)

   ![image-20250108112249665](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250108112249665.png)

2. 在`init.py`文件中引用。

   ![image-20250108112346046](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250108112346046.png)

3. 在`task.py`文件中引用。

   ![image-20250108112439566](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250108112439566.png)

4. 新增配置文件

   ![image-20250108112724144](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250108112724144.png)

### 模型改进（选看）

本次的给大家提供好的模型改进主要围绕两个方面展开，一个方面是通过添加注意力机制增加模型的精度，一个方面是通过引入一些轻量化的卷积模块降低模型的计算量。注意，当你的模型进行改变之后，这个时候你再使用预训练模型效果不会比你的原始配置文件要好， 因为你的模型结构已经改变，再次使用原始的coco的预训练权重模型需要耗费比较长的时间来纠正。所以，我们进行对比实验的时候要统一都不使用预训练模型。或者说你可以先在coco数据集上对你的改进模型进行第一个阶段的训练，然后基于第一个阶段训练好的权重进行迁移学习。后者的方式代价较大，需要你有足够的卡来做，对于我们平民玩家而言，进行第二种就蛮好。

* 准确率方面的改进

  准确率方面改进2-CBAM: Convolutional Block Attention Module

  论文地址：[[1807.06521\] CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)

  ![image-20250111194812619](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250111194812619.png)

  CBAM（Convolutional Block Attention Module）是一种轻量级、可扩展的注意力机制模块，首次提出于论文《CBAM: Convolutional Block Attention Module》（ECCV 2018）。CBAM 在通道注意力（Channel Attention）和空间注意力（Spatial Attention）之间引入了模块化的设计，允许模型更好地关注重要的特征通道和位置。

  CBAM 由两个模块组成：

  **通道注意力模块 (Channel Attention Module)**: 学习每个通道的重要性权重，通过加权增强重要通道的特征。

  **空间注意力模块 (Spatial Attention Module)**: 学习空间位置的重要性权重，通过加权关注关键位置的特征。

  该模块的代码实现如下：

  ```python
  import torch
  import torch.nn as nn
  
  class ChannelAttention(nn.Module):
      def __init__(self, in_channels, reduction=16):
          """
          通道注意力模块
          Args:
              in_channels (int): 输入通道数
              reduction (int): 缩减比例因子
          """
          super(ChannelAttention, self).__init__()
          self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
          self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
  
          self.fc = nn.Sequential(
              nn.Linear(in_channels, in_channels // reduction, bias=False),
              nn.ReLU(inplace=True),
              nn.Linear(in_channels // reduction, in_channels, bias=False)
          )
          self.sigmoid = nn.Sigmoid()
  
      def forward(self, x):
          batch, channels, _, _ = x.size()
  
          # 全局平均池化
          avg_out = self.fc(self.avg_pool(x).view(batch, channels))
          # 全局最大池化
          max_out = self.fc(self.max_pool(x).view(batch, channels))
  
          # 加和后通过 Sigmoid
          out = avg_out + max_out
          out = self.sigmoid(out).view(batch, channels, 1, 1)
  
          # 通道加权
          return x * out
  
  
  class SpatialAttention(nn.Module):
      def __init__(self, kernel_size=7):
          """
          空间注意力模块
          Args:
              kernel_size (int): 卷积核大小
          """
          super(SpatialAttention, self).__init__()
          self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
          self.sigmoid = nn.Sigmoid()
  
      def forward(self, x):
          # 通道维度求平均和最大值
          avg_out = torch.mean(x, dim=1, keepdim=True)
          max_out, _ = torch.max(x, dim=1, keepdim=True)
          combined = torch.cat([avg_out, max_out], dim=1)  # 拼接
  
          # 卷积处理
          out = self.conv(combined)
          out = self.sigmoid(out)
  
          # 空间加权
          return x * out
  
  
  class CBAM(nn.Module):
      def __init__(self, in_channels, reduction=16, kernel_size=7):
          """
          CBAM 模块
          Args:
              in_channels (int): 输入通道数
              reduction (int): 缩减比例因子
              kernel_size (int): 空间注意力卷积核大小
          """
          super(CBAM, self).__init__()
          self.channel_attention = ChannelAttention(in_channels, reduction)
          self.spatial_attention = SpatialAttention(kernel_size)
  
      def forward(self, x):
          # 通道注意力模块
          x = self.channel_attention(x)
          # 空间注意力模块
          x = self.spatial_attention(x)
          return x
  ```

* 速度方面的改进 

  速度方面改进2-GhostConv

  **Ghost Convolution** 是一种轻量化卷积操作，首次提出于论文《GhostNet: More Features from Cheap Operations》（CVPR 2020）。GhostConv 的核心思想是利用便宜的操作生成额外的特征图，以减少计算复杂度和参数量。、

  GhostConv的核心思想如是，卷积操作会生成冗余的特征图。许多特征图之间存在高相关性。GhostConv 的目标是通过减少冗余特征图的计算来加速网络的推理。GhostConv 的结构如下：

  ![image-20250109220155390](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250109220155390.png)

  **主特征图**: 使用标准卷积生成一部分特征图。

  **副特征图**: 从主特征图中通过简单的线性操作（如深度卷积）生成。

  代码实现如下：

  ```python
  import torch
  import torch.nn as nn
  
  class GhostConv(nn.Module):
      def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ratio=2, dw_kernel_size=3):
          """
          Ghost Convolution 实现
          Args:
              in_channels (int): 输入通道数
              out_channels (int): 输出通道数
              kernel_size (int): 卷积核大小
              stride (int): 卷积步幅
              padding (int): 卷积填充
              ratio (int): 副特征与主特征的比例
              dw_kernel_size (int): 深度卷积的卷积核大小
          """
          super(GhostConv, self).__init__()
          self.out_channels = out_channels
          self.primary_channels = out_channels // ratio  # 主特征图通道数
          self.ghost_channels = out_channels - self.primary_channels  # 副特征图通道数
  
          # 主特征图的标准卷积
          self.primary_conv = nn.Conv2d(
              in_channels, self.primary_channels, kernel_size, stride, padding, bias=False
          )
          self.bn1 = nn.BatchNorm2d(self.primary_channels)
  
          # 副特征图的深度卷积
          self.ghost_conv = nn.Conv2d(
              self.primary_channels, self.ghost_channels, dw_kernel_size, stride=1,
              padding=dw_kernel_size // 2, groups=self.primary_channels, bias=False
          )
          self.bn2 = nn.BatchNorm2d(self.ghost_channels)
  
          self.relu = nn.ReLU(inplace=True)
  
      def forward(self, x):
          # 主特征图
          primary_features = self.primary_conv(x)
          primary_features = self.bn1(primary_features)
  
          # 副特征图
          ghost_features = self.ghost_conv(primary_features)
          ghost_features = self.bn2(ghost_features)
  
          # 合并主特征图和副特征图
          output = torch.cat([primary_features, ghost_features], dim=1)
          output = self.relu(output)
  
          return output
  ```

### 
