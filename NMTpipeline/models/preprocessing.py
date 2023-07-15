

import tensorflow as tf
import tensorflow_text as tf_text
from custom.bertpreprocess import BertTokenizerModule
from models import constants

MAX_TOKENS = constants.MAX_TOKENS


#需要注意tf_text.pad_model_inputs 输出是两个值
def preprocessing_fn(inputs):
    with tf.init_scope():
        en_tokenizer=BertTokenizerModule(constants.target_vocab_path)
        spa_tokenizer=BertTokenizerModule(constants.context_vocab_path)
    spa = tf_text.normalize_utf8(inputs['context'],constants.normalize_utf8_method)
    #注意：由于spa (None, 1),而tokenize输入需要一维，所以需要降维。
    spa = tf.squeeze(spa, axis=1)
    spa = spa_tokenizer.tokenize(spa)
    spa, _ = tf_text.pad_model_inputs(spa,max_seq_length=MAX_TOKENS)
    
    #由于这里的预处理需要同一输出序列长度一致（如'context_in'）需要使所有序列长度一致（一个样本单词个数）
    #如果不在pipeline中，使用tf.data.Dataset可以让每个batch（batch中还是要一致的）的序列长度不一致
    #因为序列长度不会影响模型的参数个数与大小。
    en = tf_text.normalize_utf8(inputs['target'],constants.normalize_utf8_method)
    en = tf.squeeze(en,axis=1)
    en = en_tokenizer.tokenize(en)
    en, _ = tf_text.pad_model_inputs(en,max_seq_length=MAX_TOKENS+1)
    en_inputs = en[:,:-1]
    en_labels = en[:,1:]
    
    
    return {
        'context_in':spa,
        'target_in':en_inputs,
        'target_out':en_labels
    }