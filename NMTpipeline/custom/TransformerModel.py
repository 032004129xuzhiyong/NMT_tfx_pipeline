
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text

from einops.layers.tensorflow import Rearrange
from custom.bertpreprocess import BertPreprocess,BertTokenizerModule

def positional_encoding(length, depth):
    """
    如果嵌入后的shape为(batch, n, d)
    lenght: n
    depth: d
    
    return: tensor (length, depth)
    """
    depth = depth/2
    
    positions = np.arange(length)[:, np.newaxis] #(seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth #(1, depth)
    
    angle_rates = 1/(10000**depths) #(1, depth)
    angle_rads = positions * angle_rates #(seq, depth)
    
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.d_model, mask_zero=True)
        
        #两维,因为query有不同的子词长度（在预处理时没有设置一样长度），所以先设置最长的length
        #但是不同子词长度不影响各层的权重形状（shape）
        self.pos_encoding = positional_encoding(length=2048,depth=self.d_model)
    def compute_mask(self,*args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    def call(self,x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :] #添加Batch维
        return x

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )
        self.last_attn_scores = attn_scores
        
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x    
    
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
        
    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layernorm(x)
        return x
    
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff, dropout_rate)
        
    def call(self,x):
        x = self.attention(x)
        x = self.ffn(x)
        return x
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self,*, vocab_size, d_model, num_layers, 
                 num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
    def call(self,x):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,*,d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.causal_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff, dropout_rate)
    def call(self,x, context):
        x = self.causal_attention(x)
        x = self.cross_attention(x, context)
        
        self.last_attn_scores = self.cross_attention.last_attn_scores
        
        x = self.ffn(x)
        return x
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self,*, vocab_size, d_model, num_layers, 
                 num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.d_model=d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.last_attn_scores = None
    def call(self,x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.dec_layers[i](x,context)
        
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x
    
class Transformer(tf.keras.Model):
    def __init__(self, input_vocab_size:int, target_vocab_size:int,
                d_model:int=256, num_layers:int=4, num_heads:int=8, dff:int=512, dropout_rate:float=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size=input_vocab_size,
                              d_model=d_model,num_layers=num_layers,
                              num_heads=num_heads,dff=dff,
                              dropout_rate=dropout_rate)
        self.decoder = Decoder(vocab_size=target_vocab_size,
                              d_model=d_model,num_layers=num_layers,
                              num_heads=num_heads,dff=dff,
                              dropout_rate=dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inputs):
        context, x = inputs
        
        context = self.encoder(context)
        x = self.decoder(x, context)
        logits = self.final_layer(x)
        
        try:
            del logits._keras_mask
        except:
            pass
        return logits

    
#学习率调度
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        
        self.d_model = d_model
        self.d_model_copy = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        return {'d_model':self.d_model_copy,'warmup_steps':self.warmup_steps}

    
#损失
def masked_loss(label, pred):
    mask = label !=0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
    )
    loss = loss_object(label, pred)
    mask = tf.cast(mask, dtype=tf.float32)
    loss *= mask
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


#准确率
def masked_accuracy(label, pred):
    mask = label !=0
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
    match = match & mask
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)

#用于tfma的类，同masked_accuracy
class CustomMaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_masked_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.match_count = self.add_weight(name='match_count', initializer='zeros')
        self.mask_count = self.add_weight(name='mask_count', initializer='zeros')
        
    def update_state(self, label, pred, sample_weight=None):
        mask = label !=0
        pred = tf.argmax(pred, axis=2)
        label = tf.cast(label, pred.dtype)
        match = label == pred
        match = match & mask
        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            match *= sample_weight
            mask *= sample_weight
        self.match_count.assign_add(tf.reduce_sum(match))
        self.mask_count.assign_add(tf.reduce_sum(mask))
        
    def result(self):
        return self.match_count / self.mask_count
    
    def reset_states(self):
        self.match_count.assign(0)
        self.mask_count.assign(0)
        

#只能翻译一个句子的测试Translator
class Translator(tf.Module):
    def __init__(self, spa_vocab_path, en_vocab_path, transformer):
        spa_vocab_path=tf.saved_model.Asset(spa_vocab_path)
        en_vocab_path=tf.saved_model.Asset(en_vocab_path)
        spa_tokenizer = BertTokenizerModule(spa_vocab_path)
        en_tokenizer = BertTokenizerModule(en_vocab_path)
        self.tokenizers = tf.Module()
        self.tokenizers.en = en_tokenizer
        self.tokenizers.spa = spa_tokenizer
        self.transformer = transformer
        
        
    def __call__(self, sentence, max_length=128):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) ==0:
            sentence = sentence[tf.newaxis] #一个句子，一维
        sentence = self.tokenizers.spa.tokenize(sentence).to_tensor() #一个句子，二维
        encoder_input = sentence
        
        start_end = self.tokenizers.en.tokenize([''])[0] #一维，包含start_token, end_token
        start = start_end[0][tf.newaxis] #一维，[start_token]
        end = start_end[1][tf.newaxis] #一维，[end_token]
        
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start) #写入一维
        
        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack()) #(1,n) 二维
            predictions = self.transformer([encoder_input, output], training=False)
            
            predictions = predictions[:,-1:, :] #(batch, 1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=-1) #(batch, 1)
            
            output_array=output_array.write(i+1, predicted_id[0]) #写入一维: [token]
            
            if tf.reduce_all(predicted_id == end):
                break
        output = tf.transpose(output_array.stack()) #(1,n) ids
        text = self.tokenizers.en.detokenize(output)[0] #() 标量
        
        words = self.tokenizers.en.lookup_id_to_word(output) #(1,n)
        
        #output输入到transformer,不能包括end_token
        self.transformer([encoder_input, output[:,:-1]], training=False)
        #(batch, num_heads, s, t) batch=1
        attention_weights = self.transformer.decoder.last_attn_scores 
        
        return text, words, attention_weights
    
    

#用于在model导出是加上translator的签名函数
#可以翻译多个句子
class TranslatorForTFX(tf.Module):
    def __init__(self, *, model, tf_transform_output, target_vocab_path, start_token, end_token, max_seq):
        #参数中包含"*"，是为了隔开位置参数，是这些参数都是关键字参数
        #如果没有"*"那么会报错: TypeError: too many positional arguments
        self.model = model
        self.max_seq = max_seq
        self.model.tft_layer_inference = tf_transform_output.transform_features_layer()
        #因为在这里创建自定义的tokenizer(内部有LookupTable)不能保存，所以换成layer
        # self.model.id_to_word_layer = tf.keras.layers.StringLookup(
        #     vocabulary = target_vocab_path,
        #     mask_token='', oov_token='[UNK]',
        #     invert = True
        # )
        self.model.BertTokenizer = tensorflow_text.BertTokenizer(target_vocab_path)
        self.target_vocab_path=tf.saved_model.Asset(target_vocab_path)
        self.start_token = start_token
        self.end_token = end_token
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,1],dtype=tf.string)
    ])
    def __call__(self, context_texts):
        """
        context_texts: 二维，因为输入到tf_transform_output.transform_features_layer()
        的张量形状是(None,1)
        """
        transformed_context = self.model.tft_layer_inference({
            'context':context_texts
        })['context_in'] #二维(batch,max_tokens)
        
        
        #batch_size = transformed_context.shape[0]
        
        #由于batch_size为None，使得tf.fill不起作用，因此使用tf.ones_like和tf.zeros_like
        #只需要输入(batch,1)形状
        next_tokens = tf.ones_like(transformed_context,dtype=tf.int64) * self.start_token
        next_tokens = next_tokens[:,:1]
        #为了判断需要翻译的句子是否全部结束
        done = tf.zeros_like(transformed_context,dtype=tf.bool)[:,:1]
        
        tokens = tf.TensorArray(tf.int64,size=0, dynamic_size=True)
        for i in tf.range(self.max_seq):
            predictions = self.model({'context_in':transformed_context,'target_in':next_tokens})#三维(batch,1,vocab_size)
            predictions = predictions[:,-1,:] #二维(batch, vocab_size)
            next_tokens = tf.random.categorical(predictions,num_samples=1) #(batch 1)
            
            done = done | (next_tokens == tf.convert_to_tensor(self.end_token,dtype=tf.int64)) #(batch 1)
            next_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), next_tokens) #(batch,1)
            
            tokens=tokens.write(i,next_tokens)
            
            if tf.reduce_all(done):
                break
        
        tokens_einops_layer = Rearrange('t batch 1 -> batch t')
        tokens=tokens.stack()
        tokens = tf.convert_to_tensor(tokens, dtype=tf.int64)
        tokens = tokens_einops_layer(tokens) #(batch, t)
        
        #words = self.model.id_to_word_layer(tokens)
        words = self.model.BertTokenizer.detokenize(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
        result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
        
        return   result#一维
            
        
    

def plot_attention_matrix(in_words, translated_words, attention_metrix):
    #in_words 一维
    #translated_words 一维
    #attention_metrix 二维
    # The model didn't generate `<START>` in the output. Skip it.
    translated_words = translated_words[1:]

    ax = plt.gca()
    ax.matshow(attention_metrix)
    ax.set_xticks(range(len(in_words)))
    ax.set_yticks(range(len(translated_words)))

    labels = [label.decode('utf-8') for label in in_words.numpy()]
    ax.set_xticklabels(
      labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_words.numpy()]
    ax.set_yticklabels(labels)
    
def plot_attention_num_heads_matrix(sentence, translator_model, tokenizers):
    translated_text, translated_words, attention_weights = translator(
        tf.constant(sentence))
    in_ids = tokenizers.spa.tokenize(tf.constant([sentence])).to_tensor()
    in_words = tokenizers.spa.lookup_id_to_word(in_ids) #二维
    
    fig = plt.figure(figsize=(16,8))
    for h, attention_matrix in enumerate(attention_weights[0]):
        ax = fig.add_subplot(2,4,h+1)
        plot_attention_matrix(in_words[0], translated_words[0], attention_matrix)
        ax.set_xlabel(f'Head {h+1}')
    plt.tight_layout()
    plt.show()    
    
class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (text,
         words,
         attention_weights) = self.translator(sentence, max_length=128)

        return text
