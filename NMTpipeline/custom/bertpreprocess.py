
from typing import Union,List,Dict,Any,Optional
import tensorflow_text as text
import tensorflow as tf
import re
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

class BertPreprocess():
    
    #用于生成词表的类函数
    @classmethod
    def generate_vocab(cls,dataset:tf.data.Dataset,vocab_size:int=8000,
                      reserved_tokens:List[str]=['[PAD]','[UNK]','[START]','[END]'],
                      bert_tokenizer_params:Dict[str,Any]={'lower_case':True},
                      learn_params:Optional[Dict[str,Any]]=None):
        vocab = bert_vocab.bert_vocab_from_dataset(
            dataset=dataset,
            vocab_size=vocab_size,
            reserved_tokens=reserved_tokens,
            bert_tokenizer_params=bert_tokenizer_params,
            learn_params=learn_params
        )
        return vocab
    
    #把词表写入文件
    @classmethod
    def write_vocab_file(cls, filepath:str, vocab:List[str]):
        with open(filepath,'w') as f:
            for token in vocab:
                print(token, file=f)
    
    
    def __init__(self,vocab_list_or_vocab_path: Union[List[str],str,tf.saved_model.Asset],num_oov_buckets: int=1):
        self.vocab_list_or_vocab_path = vocab_list_or_vocab_path
        #read vocab_list
        if isinstance(vocab_list_or_vocab_path, str) or isinstance(vocab_list_or_vocab_path, tf.saved_model.Asset):
            #filepath
            self.vocab_list_or_vocab_path=tf.strings.split(tf.io.read_file(vocab_list_or_vocab_path))
            #self.vocab_list_or_vocab_path=tf.strings.strip(tf.io.gfile.GFile(vocab_list_or_vocab_path).readlines())
        self._VOCAB_SIZE = tf.size(self.vocab_list_or_vocab_path,out_type=tf.int64)
        #make lookup_table
        self.lookup_table=tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=self.vocab_list_or_vocab_path,
                key_dtype=tf.string,
                values=tf.range(self._VOCAB_SIZE,dtype=tf.int64),
                value_dtype=tf.int64
            ),
            num_oov_buckets=num_oov_buckets
        )

        #tokenizer
        self.tokenizer = text.BertTokenizer(self.lookup_table, token_out_type=tf.int64)
        
    def get_tokenizer(self):
        return self.tokenizer
    def get_lookup_table(self):
        return self.lookup_table
    def get_vocab(self):
        return self.vocab_list_or_vocab_path
    def get_vocab_size(self):
        return self._VOCAB_SIZE
    
    def reduce_cleanup_text_input_rag_2D(self,rag_2D_text,
        reserved_to_clean_words: List[str]=['[START]','[END]','[MASK]']):
        
        bad_words = [re.escape(word) for word in reserved_to_clean_words if word!='[UNK]']
        bad_pattern = "|".join(bad_words)
        
        bad_cells = tf.strings.regex_full_match(rag_2D_text,bad_pattern)
        result = tf.ragged.boolean_mask(rag_2D_text, ~bad_cells)
        
        result = tf.strings.reduce_join(result, separator=' ', axis=-1)
        return result
    
    def add_start_end_id_input_rag_2D(self,ragged, start_text: str='[START]',
            end_text: str='[END]'):
        count = ragged.bounding_shape()[0]
        _START_TOKEN = self.lookup_table.lookup(tf.constant(start_text))
        _END_TOKEN = self.lookup_table.lookup(tf.constant(end_text))
        starts = tf.fill([count,1],_START_TOKEN)
        ends = tf.fill([count,1],_END_TOKEN)
        return tf.concat([starts, ragged, ends], axis=1)
    
    def tokenize(self,strings, start_text: str='[START]',
            end_text: str='[END]'):
        ids = self.tokenizer.tokenize(strings)
        ids = ids.merge_dims(-2,-1)
        ids = self.add_start_end_id_input_rag_2D(ids,start_text,end_text)
        return ids
    
    def detokenize(self, id_input_2D, reserved_to_clean_words: List[str]=['[START]','[END]','[MASK]']):
        words = self.tokenizer.detokenize(id_input_2D)
        return self.reduce_cleanup_text_input_rag_2D(words, reserved_to_clean_words)
    
    def lookup_word_to_id(self,words):
        return self.lookup_table.lookup(tf.constant(words))
    
    def lookup_id_to_word(self,ids):
        return tf.gather(self.vocab_list_or_vocab_path, ids)
    
    #用于生成bert模型的输入
    def bert_preprocess(self,
            inputs: Dict[str,Any],
            col_names: List[str],
            max_seq_length: int=8,
            max_selections_per_batch: int=5,
            start_text: str='[START]',
            end_text: str='[END]',
            unk_text: str='[UNK]',
            mask_text: str='[MASK]',
            selection_rate: float=0.2,
            mask_token_rate: float=0.8,
            random_token_rate: float=0.1):
        
        ##begin
        segments = [ self.tokenizer.tokenize(inputs[name]).merge_dims(1,-1) for name in col_names]
    
        #trimmer
        trimmer = text.RoundRobinTrimmer(max_seq_length=max_seq_length)
        trimmed_segments = trimmer.trim(segments)

        _START_TOKEN = self.lookup_table.lookup(tf.constant(start_text))
        _END_TOKEN = self.lookup_table.lookup(tf.constant(end_text))
        _UNK_TOKEN = self.lookup_table.lookup(tf.constant(unk_text))
        _MASK_TOKEN = self.lookup_table.lookup(tf.constant(mask_text))
        #combine
        segments_combined, segment_ids = text.combine_segments(
            segments=trimmed_segments,
            start_of_sequence_id=_START_TOKEN,
            end_of_segment_id=_END_TOKEN
        )

        #random selection
        random_selector = text.RandomItemSelector(
            max_selections_per_batch=max_selections_per_batch, #每个batch最多几个被选中
            selection_rate=selection_rate,
            unselectable_ids=[_START_TOKEN,_END_TOKEN,_UNK_TOKEN]
        )

        #make value
        mask_values_chooser = text.MaskValuesChooser(
            vocab_size=self._VOCAB_SIZE,
            mask_token=_MASK_TOKEN,
            mask_token_rate=mask_token_rate, #使用mask_token作为替换值
            random_token_rate=random_token_rate #随机一个单词作为替换值
        )

        #random_selection and make_value
        masked_input_ids, masked_lm_positions, masked_lm_ids = (
            text.mask_language_model(
                segments_combined,
                random_selector,
                mask_values_chooser,
            )
        )

        # Prepare and pad combined segment inputs
        input_word_ids, input_mask = text.pad_model_inputs(
        masked_input_ids, max_seq_length=max_seq_length)
        input_type_ids, _ = text.pad_model_inputs(
        segment_ids, max_seq_length=max_seq_length)

        # Prepare and pad masking task inputs
        masked_lm_positions, masked_lm_weights = text.pad_model_inputs(
        masked_lm_positions, max_seq_length=max_selections_per_batch)
        masked_lm_ids, _ = text.pad_model_inputs(
        masked_lm_ids, max_seq_length=max_selections_per_batch)
        
        model_inputs = {
          #前三个等shape
          "input_word_ids": input_word_ids,
          "input_mask": input_mask,
          "input_type_ids": input_type_ids,
          #后三个等shape
          "masked_lm_ids": masked_lm_ids,
          "masked_lm_positions": masked_lm_positions,
          "masked_lm_weights": masked_lm_weights,
        }
        
        return model_inputs
    

#用于预处理的 Module
#实践导出自定义的 tf.Module，除了初始化函数，其他函数没有输入签名，导出后在导入不能使用
#tf.saved_model.save
#tf.saved_model.load

#总结：输入签名只能用Tensorflow的ragged/sparsetensor/tensor/tensorarray/确定key个数与value格式（同前）的dict
#由于 Module 除了初始化函数，其他函数用于Tensor，所以这些函数大多只有一个输入，配置尽量在初始化函数（__init__）中
#其他函数尽量不用输入配置，只输入需要转换的Tensor
#如果使用dict作为输入，需要格式如：{'key1':tf.TensorSpec(...),'key2':tf.RaggedTensorSpec(...)}，这是确定的key的个数和key名
#如果在保存这个Module同时保存vocab(以免找不到vocab)，那么可以修改只接受vocab_path作为初始化参数，然后调用
#tf.saved_model.Asset(vocab_path)，这样保存是自动复制vocab文件到模型文件中。
class BertTokenizerModule(tf.Module):
    def __init__(self,vocab_list_or_vocab_path:Union[str,List[str]],num_oov_buckets:int=1,
                 reserved_to_clean_words:List[str]=['[START]','[END]','[MASK]'],
                 start_text:str='[START]',end_text:str='[END]'):
        self.reserved_to_clean_words=reserved_to_clean_words
        self.vocab_list_or_vocab_path = vocab_list_or_vocab_path
        #read vocab_list
        if isinstance(vocab_list_or_vocab_path, str) or isinstance(vocab_list_or_vocab_path, tf.saved_model.Asset):
            #filepath
            self.vocab_list_or_vocab_path=tf.strings.split(tf.io.read_file(vocab_list_or_vocab_path))
        
        self._VOCAB_SIZE = tf.size(self.vocab_list_or_vocab_path,out_type=tf.int64)
        #make lookup_table
        self.lookup_table=tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=self.vocab_list_or_vocab_path,
                key_dtype=tf.string,
                values=tf.range(self._VOCAB_SIZE,dtype=tf.int64),
                value_dtype=tf.int64
            ),
            num_oov_buckets=num_oov_buckets
        )
        
        #start end token
        self._START_TOKEN = self.lookup_table.lookup(tf.constant(start_text))
        self._END_TOKEN = self.lookup_table.lookup(tf.constant(end_text))
        #tokenizer
        self.tokenizer = text.BertTokenizer(self.lookup_table, token_out_type=tf.int64)
        
        #signature
        self.reduce_cleanup_text_input_rag_2D.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None,None], dtype=tf.string))
        self.reduce_cleanup_text_input_rag_2D.get_concrete_function(
            tf.TensorSpec(shape=[None,None], dtype=tf.string, name='rag_2D_text'))
        
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None,None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None,None], dtype=tf.int64, name='id_input_2D'))
        
        self.lookup_word_to_id.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None,None], dtype=tf.string))
        self.lookup_word_to_id.get_concrete_function(
            tf.TensorSpec(shape=[None,None], dtype=tf.string, name='words'))
        
        self.lookup_id_to_word.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None,None], dtype=tf.int64))
        self.lookup_id_to_word.get_concrete_function(
            tf.TensorSpec(shape=[None,None], dtype=tf.int64, name='ids'))
        
    @tf.function
    def get_vocab(self):
        return self.vocab_list_or_vocab_path
    
    @tf.function
    def get_vocab_size(self):
        return self._VOCAB_SIZE
    
    @tf.function
    def reduce_cleanup_text_input_rag_2D(self,rag_2D_text):
        
        bad_words = [re.escape(word) for word in self.reserved_to_clean_words if word!='[UNK]']
        bad_pattern = "|".join(bad_words)
        
        bad_cells = tf.strings.regex_full_match(rag_2D_text,bad_pattern)
        result = tf.ragged.boolean_mask(rag_2D_text, ~bad_cells)
        
        result = tf.strings.reduce_join(result, separator=' ', axis=-1)
        return result
    
    @tf.function(input_signature=[
        tf.RaggedTensorSpec(shape=[None,None], dtype=tf.int64),
    ])
    def add_start_end_id_input_rag_2D(self,ragged):
        count = ragged.bounding_shape()[0]
        starts = tf.fill([count,1],self._START_TOKEN)
        ends = tf.fill([count,1],self._END_TOKEN)
        return tf.concat([starts, ragged, ends], axis=1)
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None],dtype=tf.string, name='strings'),
    ])
    def tokenize(self,strings):
        ids = self.tokenizer.tokenize(strings)
        ids = ids.merge_dims(-2,-1)
        ids = self.add_start_end_id_input_rag_2D(ids)
        return ids
    
    @tf.function
    def detokenize(self, id_input_2D):
        words = self.tokenizer.detokenize(id_input_2D)
        return self.reduce_cleanup_text_input_rag_2D(words)
    
    @tf.function
    def lookup_word_to_id(self,words):
        return self.lookup_table.lookup(words)
    
    @tf.function
    def lookup_id_to_word(self,ids):
        return tf.gather(self.vocab_list_or_vocab_path, ids)
        
