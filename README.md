# NMT_tfx_pipeline
这是一个简单Transformer结构的TFX-pipeline

# 写在前面

+ NMT教程来自[https://tensorflow.google.cn/text/tutorials/transformer](https://tensorflow.google.cn/text/tutorials/transformer)<br/>
+ 数据集来自教程[https://tensorflow.google.cn/text/tutorials/nmt_with_attention](https://tensorflow.google.cn/text/tutorials/nmt_with_attention)中的链接[http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip](http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip)。或者已经下载解压缩在该项目目录`spa-eng`中<br/>
+ 任务：从Spanish翻译为English<br/>
+ 运行：进入项目目录(`NMTpipeline`)，然后直接`python local_runner.py`
+ 环境：见`requirement.yaml`，需要tfx(1.13.0)可能旧版本会有不兼容，api不稳定。
+ 项目目录介绍
    + `custom`目录包含定义的预处理（教程来自[https://tensorflow.google.cn/text/guide/subwords_tokenizer](https://tensorflow.google.cn/text/guide/subwords_tokenizer)..等）和Transformer模型
    + `data`目录包含pipeline需要的输入数据，下面的`创建pipeline数据`就是生成这个
    + `models`目录包含pipeline需要的预处理和模型训练代码。
        + `constants.py` 定义预处理和模型训练参数
        + `model.py` 定义pipeline需要进行的模型训练步骤
        + `preprocessing.py` 定义预处理步骤
    + `pipeline`目录包含Pipeline的配置参数和整个pipeline定义
        + `configs.py` 定义pipeline参数
        + `pipeline.py` 定义pipeline的组件，以及模型验证的配置
    + `spa-eng`目录包含原始的数据集，pipeline不需要
    + `tfx_metadata`目录是运行pipeline后自动生成的元数据目录
    + `tfx_pipeline_output`目录是运行pipeline后自动生成的组件输出
    + `vocab`目录包含生成的词表
    + `local_runner.py`用于运行pipeline的python文件,运行直接`python local_runner.py`
    + `moduletest.ipynb`就是本文件
    + `requirement.yaml`就是程序运行的环境(tfx:1.13.0)，由conda导出，环境中有个包`model-card-toolkit`有冲突，可以不用。
+ 注意：
    + 每次生成vocab大小都不一样，需要修改`models`目录下的`constants.py`中的词表大小。
    + 这个pipeline是在本地运行的。
    + pipeline运行多次后，`tf_pipelie_output`可能变得很大，它包含每次各个组件的结果。如果不需要，整个删除
    + `tf_pipelie_output`运行结果可以结合各个组件对应的库（如tfdv、tft、tfma、TF-serving）导入结果，可视化结果，部署模型等。
    + 运行前，最好修改一下模型参数，由于笔者个人电脑限制，模型大小调小。如果资源足够，可以`d_model`和`dff`翻倍，`num_layers`为8，`batch`调大
    + 导出的模型签名（示例都在下面）：
        + `serving_default`签名函数，需要原始输入**序列化**后的examples数据，是为了用于`Evaluator`组件评估（一般模型输出就是我们需要的，但是这里不是）。
        + `transform_features`是预处理的签名函数，需要原始输入**序列化**后的examples数据。
        + `translator`是用于翻译的签名函数，需要原始输入**序列化**后的examples数据。（当然也可以改为Tensor输入）
        + `train_step`是用于继续训练的`train_step`的签名函数，需要输入原始输入的**Tensor**数据（比较方便）。一次输入一个batch的数据。当然也可以直接加载原始数据，然后用`transform_features`签名函数预处理（先batch再预处理），最后用没有任何签名的模型（也就是刚加载的模型：它的输入是经过预处理的数据）使用`fit`方法训练(要有`fit`方法需要用第二种导入)
    + 该pipeline还可以扩展或优化，比如
        + 添加Tuner组件，进行超参数调优
        + 增加保存点，进行断点续训
        + 将`translator`输入签名变为原始输入Tensor，就不用再序列化。
        + 在运行pipeline前，正确定义`Schema`然后将其路径作为`create_pipeline`函数的参数`schema_path`，这样可以多个`ExampleValidator`数据验证组件，提前观测数据漂移，训练-服务偏斜，其他异常等。
        + 添加kubeflow的config配置(需要能访问到整个项目文件，比如将这个文件放到云存储桶中或绑定一个持久卷声明，建议查看tfx的template示例)，运行后生成pipeline的压缩文件，然后可以上传到kubeflow的pipeline上运行。（需要能访问外网，因为`workflow`需要`gcr`的镜像，镜像很大）
