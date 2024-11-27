"""Script for training the TrOCR model using the provided dataset.

This script handles the model training pipeline, including data loading, 
batch preparation, model optimization, and evaluation, to fine-tune the TrOCR 
model for OCR tasks.
"""
import glob as glob
from typing import Dict

import evaluate
import torch
from transformers import (
    default_data_collator,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from src.configs import TrainingConfig, ModelConfig


class MetricComputer:
  """A class for computing Character Error Rate (CER) metrics for OCR predictions.

  Attributes:
    processor:
      The TrOCR processor used to decode model predictions into text.
    cer_metric:
      An object or function for calculating the Character Error Rate (CER), 
      which measures the accuracy of OCR predictions by comparing them to the 
      ground truth text.
  """

  def __init__(self, processor: TrOCRProcessor) -> None:
    """Initialize the metric computation object.

    Args:
      processor:
        The TrOCR processor used to decode model predictions into text 
        for evaluation against the ground truth.
    """
    self._processor = processor
    self._cer_metric = evaluate.load('cer')

  def compute_cer(self, pred) -> Dict[str, float]:
    """Compute the Character Error Rate (CER) metric for the given predictions.

    Returns:
      dict:
        A dictionary with the computed CER metric, representing the average 
        character-level error rate across the predictions.
    """
    pred_ids = pred.predictions
    labels_ids = pred.label_ids

    pred_str = self._processor.batch_decode(
        pred_ids,
        skip_special_tokens=True,
    )
    label_str = self._processor.batch_decode(
        labels_ids,
        skip_special_tokens=True,
    )

    cer = self._cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


def main() -> None:
  """The main function for training the TrOCR model using the provided dataset.

  This function manages the entire training process, including loading the dataset, 
  configuring the model and evaluating performance. The model is fine-tuned on 
  the given dataset to improve its performance on OCR tasks.
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  processor = TrOCRProcessor.from_pretrained(
      ModelConfig.MODEL_NAME,
      clean_up_tokenization_spaces=True,
      char_level=True,
  )

  model = VisionEncoderDecoderModel.from_pretrained(ModelConfig.MODEL_NAME)
  model.to(device)

  # Set special tokens used for creating the decoder_input_ids from the labels.
  model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
  model.config.pad_token_id = processor.tokenizer.pad_token_id

  # Set Correct vocab size.
  model.config.vocab_size = model.config.decoder.vocab_size
  model.config.eos_token_id = processor.tokenizer.sep_token_id
  model.generation_config.max_new_tokens = 4
  model.generation_config.early_stopping = True
  model.generation_config.num_beams = 3

  # Running `torch.load` with `weights_only` set to `False` can result in
  # arbitary code execution. Do it only if you got the file from a trusted source.
  # https://pytorch.org/docs/stable/generated/torch.load.html
  train_dataset = torch.load(
      TrainingConfig.paths.TRAIN_DATASET,
      weights_only=False,
  )
  valid_dataset = torch.load(
      TrainingConfig.paths.VALID_DATASET,
      weights_only=False,
  )

  metric_computer = MetricComputer(processor)

  training_args = Seq2SeqTrainingArguments(
      output_dir=TrainingConfig.paths.CHECKPOINTS,
      per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
      per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
      num_train_epochs=TrainingConfig.EPOCHS,
      logging_strategy='epoch',
      save_strategy='epoch',
      save_total_limit=TrainingConfig.SAVE_CKPT_LIMIT,
      report_to='tensorboard',
      eval_strategy='epoch',
      predict_with_generate=True,
      learning_rate=TrainingConfig.LEARNING_RATE,
      load_best_model_at_end=True,
      metric_for_best_model='cer',
      greater_is_better=False,
      fp16=True,
      warmup_ratio=0.1,
  )

  trainer = Seq2SeqTrainer(
      model=model,
      tokenizer=processor.tokenizer,
      args=training_args,
      compute_metrics=metric_computer.compute_cer,
      train_dataset=train_dataset,
      eval_dataset=valid_dataset,
      data_collator=default_data_collator,
      callbacks=[
          EarlyStoppingCallback(
              early_stopping_patience=TrainingConfig.EARLY_STOPPING_PATIENCE,
              early_stopping_threshold=TrainingConfig.EARLY_STOPPING_THRESHOLD,
          )
      ],
  )

  trainer.train()

  trainer.model.save_pretrained(TrainingConfig.paths.BEST_MODEL)
  processor.tokenizer.save_pretrained(TrainingConfig.paths.BEST_MODEL)
