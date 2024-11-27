"""Script for preprocessing the dataset to prepare it for TrOCR model training.

This includes tasks such as formatting, and converting data into 
a structure compatible with the model's requirements.
"""
import os

import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

from src.configs import PreprocessConfig, ModelConfig


class CustomOCRDataset(Dataset):
  """Create a custom OCR dataset for TrOCR model training, validation, and testing.

  Attributes:
    _df:
      The dataset containing the input data and corresponding labels in 
      pandas DataFrame format.
    _processor:
      The TrOCR processor used for tokenizing text and preprocessing input data.
    _train_transforms:
      A set of transformations applied to the images during the training process 
      to augment or normalize the data.
  """

  def __init__(
      self,
      df: dict,
      processor: TrOCRProcessor,
  ) -> None:
    """Initialize the instance with the dataset and processor.

    This method converts the input dataset into a pandas DataFrame, configures the 
    TrOCR processor for tokenization, and sets up the image transformations.

    Args:
      df:
        The input dataset, which will be converted into a pandas DataFrame 
        for further processing.
      processor:
        The TrOCR processor used for tokenizing text and preprocessing input data.
    """
    self._df = pd.DataFrame(df)
    self._processor = processor
    self._train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    ])

  def __len__(self) -> int:
    """Get the number of samples in the dataset.

    Returns:
      int: The total number of samples in the dataset.
    """
    return len(self._df)

  def __getitem__(self, idx) -> dict:
    """Retrieve and process a single item from the dataset at the specified index.

    This method fetches an image and its corresponding text label from the dataset, 
    applies the defined transformations to the image, and processes both the image 
    and text label using the TrOCR processor. The resulting data is formatted for 
    model training.

    Args:
      idxs:
        The index of the item to retrieve from the dataset.

    Returns:
      dict:
        A dictionary with the following keys:
        - `pixel_values`: The transformed image tensor prepared 
                          for input to the model.
        - `labels`: The tokenized text labels converted into a 
                    tensor format for training.
    """
    filename = self._df['filename'][idx]
    label = self._df['label'][idx]

    # Read the image, apply augmentations, and get the transformed pixels.
    image = Image.open(os.path.join(
        PreprocessConfig.paths.CAPTCHAS,
        filename,
    )).convert('RGB')
    image = self._train_transforms(image)

    pixel_values = self._processor(image, return_tensors='pt').pixel_values
    # Pass the text through the tokenizer and get the labels, i.e. tokenized labels.
    labels = [self._processor.tokenizer.convert_tokens_to_ids(c) for c in label]

    encoding = {
        "pixel_values": pixel_values.squeeze(),
        "labels": torch.tensor(labels)
    }

    return encoding


def main():
  """The main function for preprocessing the dataset.

  This function performs the following steps:
  1. Loads the dataset from the specified source.
  2. Splits the dataset into training, validation, and testing subsets.
  3. Converts training and validation subset into a custom TrOCR-compatible dataset.
  4. Saves the processed datasets to their respective directories.

  This prepares the data for efficient and effective training, validation, and 
  testing of the TrOCR model.
  """
  # define the processor
  processor = TrOCRProcessor.from_pretrained(
      ModelConfig.MODEL_NAME,
      clean_up_tokenization_spaces=True,
      char_level=True,
  )

  # load the filenames in the captcha directory
  image_files = np.array([
      f for f in os.listdir(PreprocessConfig.paths.CAPTCHAS)
      if f.endswith(".png")
  ])
  np.random.shuffle(image_files)
  image_files = image_files.tolist()

  # calculate the size of each dataset
  total_size = len(image_files)
  train_size = int(total_size * PreprocessConfig.TRAIN_RATIO)
  valid_size = int(total_size * PreprocessConfig.VALID_RATIO)

  # split the dataset
  train_dataset = image_files[:train_size]
  valid_dataset = image_files[train_size:train_size + valid_size]
  test_dataset = image_files[train_size + valid_size:]

  # label the dataset
  train_dataset = {
      "filename": train_dataset,
      "label": [os.path.splitext(f)[0] for f in train_dataset]
  }
  valid_dataset = {
      "filename": valid_dataset,
      "label": [os.path.splitext(f)[0] for f in valid_dataset]
  }
  test_dataset = {
      "filename": test_dataset,
      "label": [os.path.splitext(f)[0] for f in test_dataset]
  }

  # turn the dataset into cusotm TrOCR dataset
  train_dataset = CustomOCRDataset(train_dataset, processor)
  valid_dataset = CustomOCRDataset(valid_dataset, processor)

  # save the datasets
  torch.save(train_dataset, PreprocessConfig.paths.TRAIN_DATASET)
  torch.save(valid_dataset, PreprocessConfig.paths.VALID_DATASET)
  sio.savemat(PreprocessConfig.paths.TEST_DATASET, test_dataset)
