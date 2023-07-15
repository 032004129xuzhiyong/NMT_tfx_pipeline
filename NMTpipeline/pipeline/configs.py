import os

PIPELINE_NAME = 'nmt3'

#指向预处理函数
PREPROCESSING_FN = 'models.preprocessing.preprocessing_fn'
#指向模型
RUN_FN = 'models.model.run_fn'
TUNER_FN = None

#总的118944个样本，batch ：32
#默认2/3到Train,1/3到Eval
#每个epoch的steps
TRAIN_NUM_STEPS = 1239
EVAL_NUM_STEPS = 1239  #1239
EPOCHS = 100

#tfma评估时，需要满足的最小准确率
#不满足则不会push，但保存的模型还有一份在Trainer组件的model下
EVAL_ACCURACY_THRESHOLD = 0.6
