
# 如何用$ 0构建SaaS

![](https://cdn-images-1.medium.com/max/2000/0*9welXXjX4Ir82gLv.)

你将如何建立一个$ 0的SaaS？

> 不花费数千美元建设产品。

去年10月，我启动了[ipdata.co](https://ipdata.co/) - 一个IP地理定位API - 以及我的第一个SaaS。我当时的计划是尽可能多地向墙上投掷，并将我的注意力集中在哪些方面。

为了保持长久的预期，我计划在每个想法上花尽可能少的时间和金钱。只有当我有至少一个付费客户时，才会把钱花在一个想法上。

在这样做的过程中，我发现许多产品的独立层次使它可以在不花费一美元的情况下实际构建整个产品。

### 使用免费服务构建您的产品

#### AWS免费套餐（未到期优惠）

![](https://cdn-images-1.medium.com/max/2000/1*xll0_e1GfGzdn2aPrn_O4A.png)

**你可以用这个做什么**

1.  **构建API**

借助API网关和AWS Lambda，您可以获得1M免费API网关调用（前12个月），1M Lambda函数执行以及永久计算每月320万秒计算！除此之外，价格相当有利，每百万API网关请求价格为3.50美元，每百万拉姆达函数调用价格为0.20美元。

请注意，如果您正在构建的产品是长时间运行的流程，那么您可能会发现Lambda非常昂贵，因为您将根据执行时间和分配给该时间段的lambda函数的资源来收取费用。

这种设置对我们来说已经成为可能，我们已经能够在全球10个地区提供25M API请求，价格超过100美元！

> 这种设置对我们来说已经成为可能，我们已经能够在全球10个地区提供25M API请求，价格超过100美元！

2. **可扩展的数据库**

Dynamodb是AWS提供的一种快速且高度可扩展的NoSQL产品。未到期的免费级别提供25 GB的存储空间，25个读取容量单位和25个写入容量单位 - 足以处理高达200M的请求！

您可以将Dynamodb用作Key-Value存储或存储JSON文档。如果你已经与Mongodb或Redis合作过，那么转向Dynamodb应该感觉很自然。

**3.用户管理（注册，通过电子邮件和短信验证登录）**

你永远不需要在你的生活中再次编写另一个用户管理系统！AWS Cognito为您提供了所有必需的功能，以允许用户注册，登录，验证其电子邮件和电话，使用MFA等等免费！

多达50 000个用户！

超过5万用户的价格与下一个50000每用户$ 0.00550分层。

![](https://cdn-images-1.medium.com/max/1600/1*yn7zPC2QJrkx7RgHhXSSjQ.png)

**4.您的HTTP网站（S3 + Cloudfront / Netlify + AWS证书管理器）**

你想要一个静态网站。

您可以使用javascript为页面添加交互级别和动态内容。但是有了一个静态网站，

-   安全
-   速度
-   成本要低得多（除非你承载了大量的大型媒体文件）

对于这个设置你会使用

**S3**

数据存储服务，可用于为简单网站提供HTML页面。您可以获得5 GB的Amazon S3标准存储，20,000个获取请求和2,000个请求，但仅限于第一年（如果您在一年之后仍然在运行，您应该可以赚钱并且可以付款）

**CloudFront的**

Cloudfront是一个内容分发网络，它缓存您的网站页面，以便后续用户可以更快地访问您的网站，而无需从S3读取。您的第一年每月可获得50 GB数据传输输出，2,000,000个HTTP和HTTPS请求。

**Amazon Cerficate Manager**为您提供无限制的完全托管SSL证书。续订是自动化的，您可以在您的S3网站前使用这些信息，也可以将其用于您可能需要SSL证书的任何其他信息。

### 或使用Netlify ...

![](https://cdn-images-1.medium.com/max/2000/1*4KCiApFlGxt-pE5FWx5dBw.png)

[Netlify](https://www.netlify.com/)非常受开发人员的欢迎，它提供全球CDN，连续部署，单击HTTPS，与git紧密集成，流量分离测试，即时缓存失效，即时回滚和无限缩放！

他们的免费套餐非常具有包容性，其中包括：

-   个人或商业项目
-   公共或私人存储库
-   自定义域的HTTPS
-   持续部署
-   表格处理
-   社区支持
-   身份服务
-   分裂测试
-   Git Gateway

### 免费发送电子邮件

![](https://cdn-images-1.medium.com/max/2000/1*yLAsoki0f1IaA3Z4AP9-9g.png)

[**Sendgrid**](https://sendgrid.com/)

AWS Cognito会处理发送您忘记的所有密码或验证电子邮件。

对于电子邮件通讯和促销电子邮件看看Sendgrid。

他们的免费层次是每天100封电子邮件和2000个营销联系人。

除了免费套餐之外，每个月需要花费9.95美元才能发送40,000封电子邮件。

[**Mailchimp**](https://mailchimp.com/)

![](https://cdn-images-1.medium.com/max/2000/1*WQpDxuriG4TJ2igMxauo1Q.png)

Mailchimp的免费永久计划允许您每月发送12,000封电子邮件，最多可容纳2,000位订阅者。

除此之外，您可以从20美元起开始无限制发送，并根据订户数量收费

![](https://cdn-images-1.medium.com/max/1600/1*vZRE7SrsHqH3NLrrYHXORQ.png)

### 条纹

[条纹](http://stripe.com/)很容易集成并开始收集订阅。

有无处不在的弹出窗口

![](https://cdn-images-1.medium.com/max/2000/1*kRubs6JznN6iUC1etrK2hA.png)

条纹签出

最近宣布（和美丽！）预先设计的结帐表格。

![](https://cdn-images-1.medium.com/max/2000/1*n6k24yAA_RD72brIz2TKvw.png)

条纹元素

看到他们都在[这里](https://stripe.github.io/elements-examples/)行动。

最好的部分; 除非你赚钱，否则你不会得到账单。

### 销售和支持免费

![](https://cdn-images-1.medium.com/max/2000/1*1QyehRcW44-JieHLcNcYiw.png)

[**漂移**](https://www.drift.com/)

您可能在互联网上看到过其中一个聊天弹出窗口。好消息！您只需在您的网站上添加一个Javascript代码段就可以了。免费！如果你需要多个人处理支持，那么你需要一个团队计划。但是如果你是一个独立制片人，那么你很好！

我已经和客户进行了数十次交谈，而这些交谈由于漂移原因不会有。当用户看到那个小部件弹出一个“我们能帮忙吗？”时，他们回应道。很容易滑入与潜在客户的对话。

我坚信，我有很多谈话直接导致了销售。

![](https://cdn-images-1.medium.com/max/2000/1*WvEzt6zXYvbfjAVTu8ErFw.png)

[**Fullstory**](https://www.fullstory.com/)

观看用户如何通过完整的会话重播与您的网站进行交互。您可以搜索所有录像，例如，用户点击购买，但没有结帐或从购物车中移除某些物品，或者确实点击了您关心的任何html元素！

这对于查看用户设备上显示的网站时发现UI错误非常有用，您还可以直接在Pro计划中查看用户控制台的错误。

### ChartMogul

![](https://cdn-images-1.medium.com/max/2000/1*NlaRFgWCzTK-5ViIb-ewTw.png)

[Chartmogul](https://chartmogul.com/)是一项订阅分析服务，可为您的业务生成常见的SaaS指标，其中包括：

-   每月循环收入（MRR）
-   MRR运动
-   年运行率（ARR）
-   试用付费转换
-   现金周转
-   搅动

它支持与Stripe，BrainTree，Paypal，Chargify等的集成。

您可以获得精美的图表，让您深入了解SaaS的工作方式。

他们的发布计划永远免费，而您的MRR不到1万美元！

除非你真的赚了超过10K美元，否则你不会被收取费用！

### 域和DNS

![](https://cdn-images-1.medium.com/max/2000/1*8Nm2Occ2mbo_ArKGKYZe-Q.png)

[Namecheap](http://namecheap.com/) - 以每年10美元的价格购买.com。

### 公司邮箱

![](https://cdn-images-1.medium.com/max/2000/1*dIA3z0Np5BQwlJi7g_x_AA.png)

[Zoho ](https://www.zoho.com/mail/) - 获得一个类似收件箱的Gmail，您可以通过youremail@yourdomain.com发送和接收电子邮件

### Freshworks套房

![](https://cdn-images-1.medium.com/max/2000/1*jMegKxuzP2LkiGqrXWNwRg.png)

[**Freshsales**](https://www.freshworks.com/freshsales-crm/)

我喜欢Freshsales！该产品非常漂亮，令人难以置信的直观使用一个真棒免费计划！

用它来;

-   管理线索和交易
-   电话/电子邮件直接来自仪表板
-   分数线索 - 他们根据您与我发现的潜在客户的互动直观地进行自动铅排序。

Freshworks套件中有不少其他产品可以在[这里](https://www.freshworks.com/)看到[。](https://www.freshworks.com/)

[**Freshbooks**](https://www.freshbooks.com/)

![](https://cdn-images-1.medium.com/max/2000/1*1daQZoWiCH4_AKmW3FTqEA.png)

Freshworks的另一款漂亮的功能性产品。这包括; 开具发票，时间追踪，同时拥有Android和iOS移动应用程序，每月仅需15美元！

[**的Freshdesk**](https://freshdesk.com/)

如果您不想将电子邮件用于支持票务结帐[Freshdesk](https://freshdesk.com/)

![](https://cdn-images-1.medium.com/max/2000/1*_r6yTxvs8eSXyEvC7Z7Hxw.png)

他们的Sprout计划免费提供基本通话，票务和无限代理！

付费计划开始于每个代理/月每月19美元。

### 无服务器

[**Zeit.co**](https://zeit.co/now)

![](https://cdn-images-1.medium.com/max/2000/1*SVKjQ6IyMCpe2YTu-L02EQ.png)

将您的Node.js应用程序部署到世界各地的多个地区！

Zeit最多可以免费部署3次。您的代码`/_src`在您的部署中可见。

付费计划10美元/月开始10个部署（并且您的代码未公开）。

您所使用的层级是根据您的收入来计算的。

[**Heroku的**](http://heroku.com/)

![](https://cdn-images-1.medium.com/max/2000/1*krpS3r9pPkctzMLtZJn68A.png)

无需安装服务器即可运行您的Python，Ruby，Node，PHP等应用程序。

Heroku的免费提供

-   休息30分钟后休息
-   自定义域名
-   使用基于帐户的免费测试小时池

512 MB RAM│1个网页/ 1个工作人员

超出计划开始时每个月7美元。

### **监控和状态页面**

![](https://cdn-images-1.medium.com/max/2000/1*Jm_4wVMIM151VzllkRb51Q.png)

使用[Hyperping.io](https://hyperping.io/)进行正常运行时间监控。

您的SaaS可能仍然是越野车。当您的产品无法访问时，在用户发送愤怒的电子邮件之前，这可以让您领先。

您可以设置短信和电子邮件通知。停机时间也从多个地区验证，以防止误报。

### 设计

如果主题能够实现，不要浪费时间为可能失败的产品编码网站。

您可以在[Themeforest上](https://themeforest.net/)以$ 20美元获得一些体面的东西。

如果您需要定制设计工作，请检查[manypixels.co](https://www.manypixels.co/) - 每月259美元的无限设计工作。如果你不经常使用它，一个好主意就是与朋友分摊成本！你们都可以为您的项目设计一个设计机构！

![](https://cdn-images-1.medium.com/max/2000/1*GF-V3vqGMEK5YIxNgrK4Og.png)

[**Flaticon**](https://www.flaticon.com/)

![](https://cdn-images-1.medium.com/max/2000/1*Zb9Djun0xPjr-LMNVboyvw.png)

[Flaticon](https://www.flaticon.com/)为美丽的平面图标。

### **蜂房**

![](https://cdn-images-1.medium.com/max/2000/1*djirI6VjYqZQbbOU9pGKbg.png)

[Apiary](https://apiary.io/)为您提供漂亮的API文档，您可以通过Swagger规格文件或手动降价自动生成API文档！

它还为您提供一个浏览器内控制台，您的用户可以在其中使用多种语言的完整工作代码示例测试您的API - 全部自动生成！

它是免费的计划，允许你访问你需要的大多数功能。但是，如果您希望自定义域名下的文档或自定义页面以与您的品牌保持一致，那么您需要注册其中一项付费计划，每月起价99美元。

以下是您可以使用的一个工作示例：

[https://ipdata.docs.apiary.io](https://ipdata.docs.apiary.io/)

### 更多免费的东西

[**AWS激活**](https://aws.amazon.com/activate/)

![](https://cdn-images-1.medium.com/max/2000/1*I_9XY95ns1Yc7ZZ5Jry-3A.png)

如果您需要在AWS上开发更多功能，请查看[AWS Activate](https://aws.amazon.com/activate/)。

建造者计划是最直接的，因为你不需要成为加速器的一部分。

有了这个，你会得到;

-   1,000美元AWS促销信用，有效期最长为2年
-   [AWS Business Essentials](https://aws.amazon.com/activate/benefits/training/#business)在线培训（价值600美元）
-   [AWS技术基础知识](https://aws.amazon.com/activate/benefits/training/#technical)在线培训（价值600美元）
-   [自定进度实验室](https://aws.amazon.com/activate/benefits/training/#labs) 80学分（价值80美元）

**微软Azure免费套餐**

![](https://cdn-images-1.medium.com/max/2000/1*zRndg_9KWvREopb0EgS-yw.png)

我对Azure和GCE的免费套餐不太了解，但也检查一下！如果没有其他的云功能（相当于AWS Lambda）。

最后，在您的bootstrapping旅程中获得灵感，这里是WPEngine的Jason Cohen的一篇演讲，内容是纯金:) [https://vimeo.com/74338272](https://vimeo.com/74338272)

我建立[ipdata.co](https://ipdata.co/)在[公众](https://wip.chat/products/747)。在Twitter上关注我[@jonathan_trev](https://twitter.com/jonathan_trev)，我分享进度和统计数据。

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTg1NTY5OTIyNl19
-->