"""Main script for executing various processes involved in training the 
pretrained TrOCR model.

Usage:
  To run preprocessing, training, and testing sequentially:
    $ python main.py

  To collect CAPTCHA images from a specific URL:
    $ python main.py -d, --dataset

  To execute only the preprocessing step:
    $ python main.py -p, --preprocess

  To execute only the training step:
    $ python main.py -t, --train

  To execute only the testing step:
    $ python main.py -s, --test

  To display the model architecture and parameter information:
    $ python main.py -i, --info

  To display the help message and usage instructions:
    $ python main.py -h, --help
"""
import argparse

import numpy as np
import torch

from src import dataset, preprocess, train, test, info

if FIXED_SEED := True:
  seed = 1
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def main() -> None:
  """
  Execute the main script for managing the training pipeline of the TrOCR model.

  This script facilitates the key processes involved in training the pretrained 
  TrOCR model, including data collection, preprocessing, training, and testing.
  """

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--dataset",
      action="store_true",
      help="create dataset by downloading CAPTCHAs from YZU's CAPTCHA URL",
  )
  parser.add_argument(
      "-p",
      "--preprocess",
      action="store_true",
      help="preprocess the dataset for training and testing the TrOCR model",
  )
  parser.add_argument(
      "-t",
      "--train",
      action="store_true",
      help="train the TrOCR model",
  )
  parser.add_argument(
      "-s",
      "--test",
      action="store_true",
      help="test the TrOCR model",
  )
  parser.add_argument(
      "-i",
      "--info",
      action="store_true",
      help=
      "print the model architecture, total parameters and trainable parameters",
  )
  args = parser.parse_args()

  if not any([args.dataset, args.preprocess, args.train, args.test, args.info]):
    args.preprocess = True
    args.train = True
    args.test = True

  if args.dataset:
    dataset.main()

  try:
    gpu_brand = torch.cuda.get_device_name(torch.cuda.current_device())
    print((f"{gpu_brand} detected, GPU will be used."))
  except:
    print("No CUDA device detected, CPU will be used.")

  if args.preprocess:
    print("Starting preprocessing...")
    preprocess.main()
    print("Preprocessing done.")

  if args.train:
    print("Starting training...")
    train.main()
    print("Training done.")

  if args.test:
    print("Starting testing...")
    test.main()
    print("Testing done.")

  if args.info:
    info.main()


if __name__ == "__main__":
  main()
