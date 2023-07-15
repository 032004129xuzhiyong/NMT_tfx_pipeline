from typing import List
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

from models import constants
from custom.TransformerModel import Transformer,TranslatorForTFX,CustomSchedule,masked_accuracy,masked_loss

def _get_tf_examples_train_signature(model, tf_transform_output):
    
    model.tft_layer_train = tf_transform_output.transform_features_layer()
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='context_tensor'), #需要设置name，在调用时使用关键字参数
        tf.TensorSpec(shape=[None], dtype=tf.string, name='target_tensor'),
    ])
    def train_tensor(contexts, targets):
        """
        contexts: 一维Tensor
        targets: 一维Tensor
        """
        contexts = tf.expand_dims(contexts,axis=1) #因为预处理的输入是(None,1)
        targets = tf.expand_dims(targets,axis=1)
        raw_features = {'context':contexts,'target':targets}
        transformed_features = model.tft_layer_train(raw_features)
        label_features = transformed_features.pop('target_out')
        metrics_dict=model.train_step((transformed_features, label_features))
        return metrics_dict
    return train_tensor


def _get_tf_examples_translator_signature(model, tf_transform_output):
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_tf(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop('target')
        context_features = tf.io.parse_example(serialized_tf_example,
                                         raw_feature_spec)
        translator=TranslatorForTFX(model=model,
                                    tf_transform_output=tf_transform_output,
                                    target_vocab_path=constants.target_vocab_path,
                                    start_token=constants.START_TOKEN,
                                    end_token=constants.END_TOKEN,
                                    max_seq=constants.MAX_SEQ)
        #因为context_features['context']  
        #Tensor("ParseExample/ParseExampleV2:0", shape=(None, 1), dtype=string)
        #总之，tf_transform_output.transform_features_layer()针对单个特征输入是(None,1)
        #如果没有改变shape，则输出也是(None,1)
        texts=translator(context_features['context'])
        return {'outputs':texts}
    return serve_tf_examples_tf


def _get_tf_examples_serving_signature(model, tf_transform_output):
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        """Returns the output to be used in the serving signature."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        # Remove label feature since these will not be present at serving time.
        raw_features = tf.io.parse_example(serialized_tf_example,
                                         raw_feature_spec)
        #需要target生成输入target_in，所以不能删除target
        transformed_features = model.tft_layer_eval(raw_features) #dict
        transformed_features.pop('target_out')
        outputs = model(transformed_features) #context_in, target_in
        return {'outputs': outputs}

    return serve_tf_examples_fn

def _get_transform_features_signature(model, tf_transform_output):
    
    model.tft_layer_tf = tf_transform_output.transform_features_layer()
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example,
                                         raw_feature_spec)
        transformed_features = model.tft_layer_tf(raw_features)
        return transformed_features
    return transform_features_fn

def _input_fn(file_pattern: List[str],
             data_accessor: tfx.components.DataAccessor,
             schema: schema_pb2.Schema,
             batch_size:int=200) -> tf.data.Dataset:
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key='target_out'),
        schema)

def _build_keras_model(d_model,num_layers,num_heads,dff,dropout_rate,warmup_steps):
    
    context_inputs = keras.layers.Input(shape=(constants.MAX_TOKENS,), name='context_in')
    target_inputs = keras.layers.Input(shape=(None,), name='target_in')
    
    transformer = Transformer(
        input_vocab_size = constants.context_vocab_size,
        target_vocab_size = constants.target_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout_rate
    )
    
    output = transformer([context_inputs, target_inputs])
    model = keras.Model(inputs=[context_inputs, target_inputs], outputs=output)
    
    learning_rate = CustomSchedule(d_model,warmup_steps=warmup_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    model.compile(loss=masked_loss,
                  optimizer=optimizer,
                  metrics=[masked_accuracy])
    
    return model

def run_fn(fn_args: tfx.components.FnArgs):
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    transformed_schema = tf_transform_output.transformed_metadata.schema
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    train_batch_size = (
        constants.TRAIN_BATCH_SIZE * mirrored_strategy.num_replicas_in_sync)
    eval_batch_size = (
        constants.EVAL_BATCH_SIZE * mirrored_strategy.num_replicas_in_sync)
    
    #dataset
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        transformed_schema,
        batch_size=train_batch_size)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        transformed_schema,
        batch_size=eval_batch_size)
    
    #model
    with mirrored_strategy.scope():
        model = _build_keras_model(constants.d_model,
                                   constants.num_layers,
                                   constants.num_heads,
                                   constants.dff,
                                   constants.dropout_rate,
                                   constants.warmup_steps)
    
    #callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_masked_accuracy',
                                                     factor=0.1,patience=3)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_masked_accuracy',
                                                  patience=5,restore_best_weights=True)
    
    model.fit(
        train_dataset.repeat(),
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset.repeat(),
        validation_steps=fn_args.eval_steps,
        epochs=fn_args.custom_config['epochs'],
        callbacks=[
            tensorboard_callback,
            reduce_lr,
            early_stop
        ]
    )
    
    signatures = {
        'serving_default':_get_tf_examples_serving_signature(model,tf_transform_output),
        'transform_features':_get_transform_features_signature(model,tf_transform_output),
        'translator':_get_tf_examples_translator_signature(model,tf_transform_output),
        'train_step':_get_tf_examples_train_signature(model, tf_transform_output)
    }
    
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
    
    

