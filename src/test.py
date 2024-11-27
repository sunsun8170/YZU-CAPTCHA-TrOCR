"""Script for evaluating the TrOCR model on a dataset of CAPTCHA images.

This script performs testing of the fine-tuned TrOCR model by processing a set 
of CAPTCHA images and comparing the model's predictions against the ground truth
labels. It generates performance metrics such as accuracy or error rates to 
assess the model's effectiveness on CAPTCHA recognition tasks.
"""
import glob as glob
import os

from matplotlib import pyplot as plt
from PIL import Image
from scipy import io as sio
import torch
from tqdm import tqdm
from transformers import (VisionEncoderDecoderModel, TrOCRProcessor)

from src.configs import TestConfig, ModelConfig


def ocr(
    image: Image,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    device: torch.device,
) -> str:
  """Perform Optical Character Recognition (OCR) on the given image.

  Args:
    image:
      The input image containing text to be recognized. This image will be
      processed for text extraction.
    processor:
      The TrOCR processor used for tokenizing the input image and decoding the 
      predicted text.
    model:
      The Vision Encoder and Text Decoder model used for performing OCR tasks, 
      including image-to-text conversion.
    device:
      The device used for model inference, either "cuda" for GPU acceleration or
      "cpu" for CPU processing.
      
  Returns:
    str:
      The text recognized from the image.
  """
  # We can directly perform OCR on cropped images.
  pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
  generated_ids = model.generate(pixel_values)
  generated_text = processor.batch_decode(
      generated_ids,
      skip_special_tokens=True,
  )[0]

  return generated_text


def bar_chart(success: int, failed: int) -> None:
  """Generate a bar chart to visualize the testing results.

  This function creates a bar chart that visually represents the number of 
  successful and failed predictions made by the model during testing and will be
  saved as a PNG file. It provides a simple way to assess model performance at a
  glance.

  Args:
    success:
      The number of successful predictions made by the model.
    failed:
      The number of failed predictions made by the model.
  """
  accuracy = (success / (success + failed)) * 100

  plt.figure(figsize=(5, 7))

  plt.bar(['success', 'failed'], [success, failed], color=['green', 'red'])
  plt.title(TestConfig.ACC_REPORT_TITLE, fontsize=14)
  plt.xlabel('Result', fontsize=12)
  plt.ylabel('Amount', fontsize=12)

  plt.text(
      -0.5,
      success + 0.5,
      f'{accuracy:.1f}%',
      fontsize=12,
      color='blue',
  )
  plt.text(
      1.2,
      failed + 0.5,
      f'{100 - accuracy:.1f}%',
      fontsize=12,
      color='blue',
  )

  plt.tight_layout()
  plt.savefig(TestConfig.paths.ACC_REPORT)


def main() -> None:
  """The main function for testing the TrOCR model on a dataset of CAPTCHA images.

  This function evaluates the performance of the TrOCR model by processing a 
  dataset of CAPTCHA images, comparing the model's predictions against the 
  ground truth, and generating relevant performance metrics to assess accuracy.
  """

  device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

  processor = TrOCRProcessor.from_pretrained(
      ModelConfig.MODEL_NAME,
      clean_up_tokenization_spaces=True,
      char_level=True,
  )
  trained_model = VisionEncoderDecoderModel.from_pretrained(
      ModelConfig.BEST_MODEL).to(device)

  test_dataset = sio.loadmat(TestConfig.paths.TEST_DATASET)

  success = 0
  failed = 0
  dataset_len = len(test_dataset["filename"])

  with tqdm(total=dataset_len, desc="Testing", position=0) as bar:
    for i in range(dataset_len):
      image_path = os.path.join(
          TestConfig.paths.CAPTCHAS,
          test_dataset["filename"][i],
      )
      image = Image.open(image_path).convert('RGB')
      image_label = test_dataset["label"][i]
      ocr_text = ocr(image, processor, trained_model, device)
      if ocr_text == image_label:
        success += 1
      else:
        failed += 1

      bar.set_description(
          f"Acc: {success / (success + failed) * 100:.2f} % | Testing")
      bar.update(1)

  bar_chart(success, failed)
