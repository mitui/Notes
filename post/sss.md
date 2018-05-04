


## NVIDIA高级系统架构师付庆平：如何搭建DGX-1高性能AI集群

_2018-04-11_ _创课_ [智东西](https://mp.weixin.qq.com/s?__biz=MzA4MTQ4NjQzMw==&mid=2652712741&idx=2&sn=7dfe40e9b6c11a4af43efe79cb539430&chksm=847db82bb30a313ddf31f93c686d73df8cf73335564e2610b4ce992bdfb68803319b61f4fd51&mpshare=1&scene=1&srcid=04113LixVXG4UC30iKflJ2YQ&pass_ticket=OaG2sZp4X%2BcApD9JJdF33yJV5pibg%2F%2FWQUKimfKeW5IwQgZMWDcEMfh42OZrNBEl##)

去年5月，在2017年度GPU技术大会（GTC）上，英伟达发布了超级计算机NVIDIA DGX Station。作为针对人工智能开发的GPU工作站，NVIDIA DGX Station的计算能力相当于400颗CPU，而所需功耗不足其1/20，而计算机的尺寸恰好能够整齐地摆放在桌侧。数据科学家可以用它来进行深度神经网络训练、推理与高级分析等计算密集型人工智能探索。

作为致力于将深度学习人工智能技术引入到智能医学诊断的系统开发商，图玛深维采用了DGX Station以及CUDA并行加速来进行神经网络模型的训练，并在此基础上开发出了σ-Discover Lung智能肺结节分析系统。σ-Discover Lung系统能够帮助医生自动检测出肺结节、自动分割病灶、自动测量参数，自动分析结节良恶性、提取影像组学信息、并对肺结节做出随访，大幅度减少结节筛查时间，减少读片工作量，提高结节的检出率，并且提供结节的良恶性定量分析，提高筛查的效果。σ-Discover Lung系统于去年8月发布。去年12月，图玛深维完成软银中国领投的2亿人民币B轮融资。

3月23日起，智东西联合NVIDIA推出「NVIDIA实战营」，共计四期。第一期由图玛深维首席科学家陈韵强和NVIDIA高级系统架构师付庆平作为主讲讲师，分别就《深度学习如何改变医疗影像分析》、《DGX超算平台-驱动人工智能革命》两个主题在智东西旗下「智能医疗」社群进行了系统讲解。目前，「NVIDIA实战营」第二期已经结束。「NVIDIA实战营」第三期将于4月13日20点开讲，主题为《智能监控场景下的大规模并行化视频分析方法》，由西安交通大学人工智能和机器人研究所博士陶小语、NVIDIA高级系统架构师易成共同在智东西「智能安防」社群主讲。

本文为NVIDIA高级系统架构师付庆平的主讲实录，正文共计3515字，预计5分钟读完。在浏览主讲正文之前，先了解下本次讲解的提纲：

-Tesla Volta GPU  
-DGX-1硬件设计及软件堆栈  
-搭建DGX-1高性能AI集群  
  
付庆平：大家好，我是付庆平，来自NVIDIA。非常感谢陈博士的讲解。我来介绍一下NVIDIA做的一些突破性的工作，以及DGX这样一个整体的解决方案。如何去帮助大家尽快地高性能地去完成自己的深度学习工作。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAdQ8R2FLCWpKB4bJGkiaIDCro0TGTIsEPMeW5CCtHKkpDzftCL5Xfd1g/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

这张图片大家可以看到，从2012年开始，大家已经不断地在进行GPU+深度学习的探索；在2015年ImageNet竞赛中，DNN（深度神经网络）图像识别的水平已经完全超越了人类；在2015年，语音识别系统也达到了超人类的语音识别水平。这主要得益于计算本身的高可靠、高性能，能够避免人在识别的过程中的一些环境，身体，心理等因素造成的失误，从而达到更高的准确率。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAkfMz7wV3xXfvia1xT9JEG5mLiccRxEDsibNibngPZvU7hOx5A4dKe8sSjg/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

这里我借鉴陈博士那张PPT，从这张PPT可以看到，在医疗诊断的过程当中，图像识别起到了非常重要的作用。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAeUHGNbYIN9ob1x5OicyRhrqLMWpVlzdFJIERP8tbyO2RRGrnh8vBGTA/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

在模型训练之前，我们的研发人员需要去准备这样一个深度神经网络。在准备好我们的网络之后，需要去训练这个网络，训练的过程需要大量的数据，通过大量数据的准备和计算，去完成我们在整个网络weights值的更新。

陈博士所提到的，每个病人可能有几百张的照片要去计算，所有的医院所有病人加起来，可能有几千万张甚至上亿张的图片需要我们去完成计算。这个过程当中，一方面我们需要优秀的算法，优秀的网络，但另一方面我们需要非常高性能的基础设施去完成这样的计算过程。

在网络训练完成之后，就得到了一个能够满足我们识别需求的深度神经网络，下一步是要把这个网络部署到我们实际的应用场景当中。在应用场景当中，需要做到如何去高效、高性能、快速地去识别图片。这里需要提到两点：

1、吞吐量，也即单位时间内所能识别的图片数量；  
2、识别的延迟。

这也是我们在训练端所需要关注的两个非常重要的因素。下面我基于上面几点来介绍NVIDIA的整体解决方案。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUATJmJPEI8gb4mCWLMCA6hicPyL8vJK9nhGT0QbdwGmVJbkHT3awY2ZXA/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

什么是NVIDIA面向HPC（高性能计算）以及深度学习的整体的解决方案呢？

首先是NVIDIA所提供的Tesla GPU，最新的V100 GPU、DGX-1等基础硬件设施，再上一层是NVIDIA所提供的SDK，我们如何更好地去应用这些高性能的硬件措施，包括CuDNN 、TensorRT等。TensorRT主要应用于推理端，CuDNN主要应用于神经网络训练。NCCL是GPU之间的集合通信库，以及其他的一些数学库。NVIDIA还提供Caffe、Caffe2、Tensorflow这些专门针对硬件进行优化的主流深度学习框架。

另外一方面，NVIDIA也为高性能计算提供完整的解决方案，我们在做深度学习优化的过程中，也在做AI的高性能计算。

我将就以下三个方面向大家介绍NVIDIA面向HPC（高性能计算）以及深度学习的整体解决方案。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAI2vfeLVNV2z3oPJaHOr1TefszcrSSAZvWhqZDEHWt5NG9xibQSt1PbA/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

V100 GPU是目前NVIDIA针对高性能计算以及深度学习所推出的最新版GPU。首先我来对NVIDIA GPU的发展做一定的介绍：

2008年，NVIDIA推出了Tesla GPU，Tesla GPU第一款就是CUDA的GPU，CUDA的出现，方便了我们科研人员在GPU上进行编程，完成自己的科研计算任务。

2010年，在Fermi GPU中，增加了双精度计算以及内存ECC相关功能的支持，每一代GPU都会有新的功能加入，其计算能力上也会有非常大的提升。

2012年，开普勒GPU,增加了动态的线性调度以及GPU Direct等功能，GPU Direct可以实现GPU之间的直接通信，对GPU间并行计算的性能有了非常大的提升。

在Pascal架构的GPU中增加了Unified Memory、3D堆叠显存，以及NVLink GPU通信的一些新型功能的支持，这些功能对加速高性能计算及人工智能的发展都起了突破性的作用。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAmRZibkEW8kRjP8117mmh2yj4oeSjoMEpT5sIibq1iaRBIUZBOIvagEERg/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

接下来我将介绍最新的 Volta架构 V100 GPU。V100 GPU相对于P100 GPU有了突破性的提升。主要为以下几点：

首先，Volta GPU在基本架构上有了非常大的改进。在计算能力不断增强的基础上，它的耗能是P100 GPU的50%，并加入了最新面向深度学习的Tensor Core专用计算单元，可以实现125 TFlOPS的运算能力；

其次，在拥有了这样一个强大的计算核心的同时，我们在GPU内部增加了高带宽显存以及NVLink这样的新型技术。面向推理端，我们提供新型的多进程服务功能，进一步增大了推理的吞吐量，降低推理延迟；

最后是单指令多数据的模型，在多进程之间增加了一个新的通信算法和功能。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAdRIMtMmB2akm8Rkeuss1zgMt5bPwpxribMVPgOwN7eUoSTEl7hMboqA/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

我首先对GPU的内部进行基本的介绍，在V100 GPU内集成了21B个晶体管，包括了80个SM流处理器，5120个CUDA Cores以及640个TensorCores。

右图中绿色的部分就是我们GPU内部的流处理器，流处理器是完成计算任务最基本的处理单元。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAwgoRZCN8P4jk8n1IstBLvjicgJxSlsKpmvBW1Butftibgz8IajfdIhEQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

一个流处理器包含了64个单精度计算单元、32个双精度计算单元、64个整型计算单元，以及8个计算能力最强的TensorCores。从图中我们也可以看到，TensorCores在一个SM流处理器中占用了很大的面积。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

从上表中我们可以看到，针对深度学习，V100的训练性能以及推理性能相对于P100都有了非常大的提升，其中训练性能提升达到了12倍。在训练的过程中会有大量的数据读取需求，V100的高带宽显存带宽达到了900GB每秒，相对于P100也有了1.2倍的提升。对于多GPU训练，GPU之间的通信带宽以及延迟起到了决定性的作用，V100相当于P100也有了1.9倍的提升。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAa08wfzqlVXyeMyBzPU1Xz6K46fxazicibPayaqEzzBPQH0AkE8qEf6rQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

下面我对大家比较关心的，也是我们计算能力最强的Tensor Core进行介绍。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAyHTr6zc2cQ5PCgIHre6IzLpbkSRk1V8MiatIjFmRTH38qU6eJvBnl8A/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

我们刚才提到Tensor Core拥有125TFLOPS的超强的计算能力，125T意味着什么呢？也就是说一台双路的服务器，它的双浮点计算能力应该在一个T左右，而加入了Tensor Core之后，GPU的运算能力可能相当于几百块CPU的计算能力。这样一个计算能力的实现，主要依赖于Tensor Core在一个时钟周期内能够实现一个四维矩阵的乘加运算，等于是针对卷积神经网络中矩阵的乘加运算的一个专用的运算单元。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAnLtxNDv7fTJ84B2d7IVAnKao7icmKK38iaB6D6MPVtichwqvkj3rVGOaA/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

那么我们如何去使用Tensor Core超强的计算能力呢？其实Tensor Core已经作为_device_函数封装在了CUDA中，同时CuBLAS、CuDNN也提供了相应的调用Tensor Core的API，在主流的深度学习框架当中，特别是NVIDIA提供的Docker版本的Caffe 、Caffe 2都已经集成了Tensor Core的使用，前提是要把CUDA版本升级到最新。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

大家可以看到，当我们使用CUDA Core之后，V100的运算能力相对P100有了将近9.3倍的提升。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

Tensor Core之外，HBM2显存在训练过程中也起到了非常关键的作用，训练过程其实是一个数据处理的过程，牵涉到大量数据的缓存，V100高带宽显存的利用率相对P100也有了1.5倍的提升。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

从这张图中可以看到V100 GPU的NVLink连接方式带宽已经达到了300GB每秒，相当于V100有了很大的提升。在单机多卡的训练场景当中起到了非常重要的作用。

下面对面向人工智能的超级计算机DGX-1的硬件设计及软件堆栈向大家进行介绍。DGX-1可以说是集成了NVIDIA从基础的硬件、SDK，到主流的深度学习框架的整体解决方案，

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAnaKq5dueAe0nfljXVVapQ4q7NBAjby8UYAib23anpHWXRfUFa2Iqia4g/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

大家可以看一下DGX整体的产品系列，包括DGX Station、DGX-1以及NVIDIA GPU的Cloud服务。

DGX Station是面向桌面端，由四张NVLink连接的GPU卡组成，采用了水冷静音的方式，非常便携，可以放到办公室里使用。DGX-1面向数据中心，使用了8块V100 GPU。GPU之间采用NVLink连接。下面我将对DGX-1硬件设计做一个详细的介绍。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAG0lxc7APlmicQXhmu6SI4BAUkWHJYko2ibfdWIuOWJsKicfL9sVlv8MgA/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

我们来看这张表，DGX-1配备了8块最新的Tesla V100 GPU，为这台服务器提供了目前业界最高的人工智能以及HPC（高性能计算）的计算能力，整个系统显存达到了128GB，同时配备了4块1.92TB的SSD RAID 0的方式提供给大家，主要是为了降低深度学习过程中读取数据的延迟。同时配备了4张IB EDR网卡，目的在于降低多机多卡训练过程中网络之间的延迟。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

DGX拥有了非常优秀的硬件设计，同时也提供了一套整体的解决方案。从这张图可以看到，DGX是基于NVIDIA Docker解决方案，在Docker容器的基础上，NVIDIA提供Caffe、TensorFlow、Theano等所有的主流深度学习框架，这些深度学习框架都是我们的研发人员针对底层的GPU硬件以及相关的SDK经过特殊优化之后的。

我们的用户拿到这些学习框架Image之后，在短时间内，一天甚至半天时间，就可以开始深度学习的计算任务。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAclyQBa0zqpucB4tNRzmiaSALDldW0RLab3hGDdxOt2ia4vA0odl18jdg/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

基于Docker解决方案，一台DGX-1超级人工智能服务器可以运行多个深度学习框架，避免了之前可能存在的一些软件版本上的冲突，进一步方便进行科研任务。从下面这张图我们可以看到在DGX Station这样一个桌面级的服务器，可以完成程序的编译、网络的测试等任务，等测试任务完成之后，可以把训练好的模型直接部署到数据中心去进行大规模的训练。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUArkFocMNfPibl7qxbDdRVVC6vcvlOTXCfJdyHZT7duEBTmyUEZjicA3yw/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

下面我将对如何运用DGX-1来搭建一套高性能的AI集群进行介绍。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAZibv8uS3WsyJ3UW4W7ePUaHwzH3XsKKmQjCmaNkAAGRe6fsN9TtFgyg/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

这张图片所显示是使用NVIDIA的DGX-1所搭建起来的一台124个节点组成的超级计算机，我们就以这个为出发点的来研究如何搭建DGX高性能超算集群。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAb0wmmFMfVN9cIYvlf2goLmsrBzUKiaALEtibG1nDtFNqHsEhMtPUS4Wg/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

在集群中，DGX-1作为基本的计算节点，节点之间的连接是通过EDR IB Switch交换机，每个节点配备了四张IB EDR的网卡，以达到最佳的训练性能。同时集群也采用了Docker解决方案，当训练好自己的模型之后，可以直接使用Docker容器的方式，部署到我们的集群当中进行训练。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

在集群搭建的过程当中，我们要考虑以下几点：

1、如何获得最高的计算能力，当然是要配备最新的高性能GPU。

2、网络如何互联，传统的高性能集群，一般是每个节点单张EDR卡。而在DGX-1集群当中，单节点都配备了四张EDR卡，实验也表明四张EDR卡，能够显著地提升训练性能。

3、存储，因为牵扯到大量数据的训练，我们拥有了Tensor Core这样一个最高的计算能力，就必须配备高性能低延迟的存储，在单台的DGX-1中我们也都配备了SSD的缓存。

4、基础设施，目前DGX-1所搭建的集群在Green500是排名第一的，这点我就不多做介绍了。

![](https://mmbiz.qpic.cn/mmbiz_png/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUAd7LZOWcsw9r6bhvqyKpFGic7Sg0Bon5MKB3kwDHMs4T0PvTwUUSyj2g/640?wx_fmt=png&wxfrom=5&wx_lazy=1)

这张图主要介绍了在DGX-1 中8卡GPU、4张MLX0 EDR卡场景下，如何实现多节点之间的通信。也可以看到我们通过PCIe进行CPU到GPU到MLX0 EDR卡之间的绑定，进一步提高训练时的通信效率。

最后我想说，DGX-1不仅仅只是一台硬件的服务器，更重要的它集成了NVIDIA整体的解决方案，包括主流的深度学习框架以及NVIDIA所能提供的一些优秀的深度学习的SDK。

今天演讲就是这些，谢谢大家。

**如需本期实战营课件及音频，可以在智东西公众号回复关键字“实战营”获取。**

----------

  

智东西联合NVIDIA推出「NVIDIA实战营」，共计4期，7位讲师参与。本周五晚8点，第三期继续免费开讲，西安交通大学人工智能和机器人研究所博士陶小语、NVIDIA高级系统架构师易成二位讲师将分别就《智能监控场景下的大规模并行视频分析方法》、《DGX-2—驱动智能监控革命》进行系统讲解。长按二维码报名（或者点击底部「阅读原文」填写主群申请表），免费入群听课。

![](https://mmbiz.qpic.cn/mmbiz_jpg/z7ZD1WagSLj1NlBFkg9gAO5nCeBqNmUASiadErYwR87iaYd923j6MqkTakibIyNfo6d7FVz1CX9NCYiahNHp8yf3eQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTY5NTYyMDQ1OSwyMDM4MzA3ODI1XX0=
-->