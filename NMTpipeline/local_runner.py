import os
from absl import logging

from tfx import v1 as tfx
from pipeline import configs
from pipeline import pipeline

OUTPUT_DIR = '.'

PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             configs.PIPELINE_NAME)
METADATA_PATH = os.path.join(OUTPUT_DIR, 'tfx_metadata', configs.PIPELINE_NAME,
                             'metadata.db')

SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def run():
    """Define a pipeline."""

    tfx.orchestration.LocalDagRunner().run(
      pipeline.create_pipeline(
          pipeline_name=configs.PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_PATH,
          preprocessing_fn=configs.PREPROCESSING_FN,
          run_fn=configs.RUN_FN,
          tuner_bool=False,
          tuner_fn=None,
          epochs=configs.EPOCHS,
          train_args=tfx.proto.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
          eval_args=tfx.proto.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
          eval_accuracy_threshold=configs.EVAL_ACCURACY_THRESHOLD,
          serving_model_dir=SERVING_MODEL_DIR,
          metadata_connection_config=tfx.orchestration.metadata
          .sqlite_metadata_connection_config(METADATA_PATH)))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()

