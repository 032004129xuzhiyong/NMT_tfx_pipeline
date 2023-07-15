
#preprocessor
#处理原文与译文时最大token数
MAX_TOKENS=20
#译文的词表路径，在启动pipeline前需要先生成词表
target_vocab_path = './vocab/en_vocab.txt'
#原文词表路径
context_vocab_path = './vocab/spa_vocab.txt'
#预处理中规范化方法，tf_text.normalize_utf8的参数之一
#规范化提高准确率
normalize_utf8_method = 'NFKD' # NFC NFKC NFD NFKD


#model
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

#建立模型需要提前准备的词表大小
#原文词表大小
context_vocab_size = 4801
#译文词表大小
target_vocab_size = 3888

#建立模型的参数
d_model=128  #嵌入向量的长度（最后一维的长度）
num_layers=4 #EncoderLayer和DecoderLayer模块各有几个
num_heads=8 #注意力头的数量
dff=256  #前馈层中隐藏层的units值（注意不是前馈层中最后一个隐藏层的units值）
dropout_rate=0.1  #所有的dropout率
warmup_steps=4000  #学习率调度，在warmup_steps步之前，学习率从0开始上升，之后学习率下降


#infer
MAX_SEQ = 128  #translator翻译时限制的最大预测步数，也就是译文最长长度
START_TOKEN = 2  #'[START]' 与 '[END]' 所属的token_id，需要先用上面生成的词表测试，不是随意设置。
END_TOKEN = 3  #同上