"""Script for displaying information about the TrOCR model.

This script loads a specified TrOCR model, prints its architecture, and displays 
information about the total number of parameters and trainable parameters. 
It is useful for debugging, verifying model configurations, and understanding 
the resource requirements of the model.
"""
import torch
from transformers import VisionEncoderDecoderModel, logging

from src.configs import ModelConfig


def main() -> None:
  """Load and display the TrOCR model's architecture and parameter details."""
  logging.set_verbosity_error()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = VisionEncoderDecoderModel.from_pretrained(ModelConfig.MODEL_NAME)
  model.to(device)

  # Print the model architecture
  print("Model Architecture:")
  print(model)

  # Calculate and print the total and trainable parameters
  total_params = sum(p.numel() for p in model.parameters())
  total_trainable_params = sum(
      p.numel() for p in model.parameters() if p.requires_grad)
  print(f"\nModel Parameter Details:")
  print(f"{total_params:,} total parameters.")
  print(f"{total_trainable_params:,} training parameters.")
