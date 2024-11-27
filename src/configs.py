"""This configuration module defines the configuration classes for the bot, model, 
preprocessing, training, and testing processes.

It provides structured settings and parameters to manage and customize the 
behavior of the bot, model training pipeline, data preprocessing, and evaluation.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BasePathConfig:
  """A class for managing the base paths used throughout the project.

  This class provides a centralized way to define and access the base directories 
  and file paths required for various operations such as dataset storage, 
  results generation, and model checkpoints. The paths are structured 
  hierarchically to ensure clarity and maintainability.

  Attributes:
    BASE:
      The base directory for the project.
    CAPTCHAS:
      The directory where downloaded CAPTCHA images are stored.
    BASE_RESULTS:
      The root directory for storing all result files and datasets.
    PREP_RESULTS:
      The directory for storing preprocessed datasets.
    TRAIN_DATASET:
      The file path for the serialized training dataset.
    VALID_DATASET:
      The file path for the serialized validation dataset.
    TEST_DATASET:
      The file path for the serialized test dataset.
    TRAIN_RESULTS:
      The directory for storing training-related results such as logs and checkpoints.
    CHECKPOINTS:
      The directory for storing model checkpoint files.
    BEST_MODEL:
      The file path for storing the best-performing model during training.
    TEST_RESULTS:
        The directory for storing results related to model testing.
    ACC_REPORT:
        The file path for storing the accuracy report visualization (e.g., a PNG chart).
  """

  BASE: Path = Path(".")

  CAPTCHAS: Path = BASE / "captcha_imgs"

  BASE_RESULTS: Path = BASE / "results"

  PREP_RESULTS: Path = BASE_RESULTS / "datasets"
  TRAIN_DATASET: str = str(PREP_RESULTS / "train_dataset.pt")
  VALID_DATASET: str = str(PREP_RESULTS / "valid_dataset.pt")
  TEST_DATASET: str = str(PREP_RESULTS / "test_dataset.mat")

  TRAIN_RESULTS: Path = BASE_RESULTS / "train"
  CHECKPOINTS: Path = TRAIN_RESULTS / "checkpoints"
  BEST_MODEL: Path = TRAIN_RESULTS / "best_model"

  TEST_RESULTS: Path = BASE_RESULTS / "test"
  ACC_REPORT: Path = TEST_RESULTS / "acc_report.png"

  def __post_init__(self):
    for attr, path in self.__dict__.items():
      if isinstance(path, Path) and not path.suffix:
        path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BotConfig:
  """A class for managing the configuration of the CAPTCHA bot.

  This class encapsulates all the settings and parameters required to configure 
  the CAPTCHA bot, including file paths, URLs, and operational limits. It uses 
  a frozen dataclass to ensure immutability, providing a consistent and 
  reliable configuration throughout the project.

  Attributes:
    paths:
      Configuration for base paths used in the project, such as directories 
      for saving CAPTCHA images.
    LOGIN_URL_REFERER:
      The URL used as the referer when logging in to the YZU course selection system.
    LOGIN_URL:
      The URL for the login endpoint of the YZU course selection system.
    CAPTCHA_URL:
      The URL for retrieving CAPTCHA images from the YZU system.
    NUM_TO_DOWNLOAD:
      The number of CAPTCHA images to download for dataset creation.
    REQUEST_TIMEOUT:
      The timeout duration (in seconds) for HTTP requests made by the bot.
  """
  paths: BasePathConfig = BasePathConfig()
  LOGIN_URL_REFERER: str = "https://isdna1.yzu.edu.tw/Cnstdsel/default.aspx"
  LOGIN_URL: str = "https://isdna1.yzu.edu.tw/CnStdSel/Index.aspx"
  CAPTCHA_URL: str = "https://isdna1.yzu.edu.tw/CnStdSel/SelRandomImage.aspx"
  NUM_TO_DOWNLOAD: int = 10000
  REQUEST_TIMEOUT: float = 5.0


@dataclass(frozen=True)
class ModelConfig:
  """A class for managing the configuration of the TrOCR model.

  This class defines the settings required for initializing and using the 
  TrOCR model, including the model name and the file path for loading the 
  best-performing model checkpoint.

  Attributes:
    MODEL_NAME:
      The name of the pretrained TrOCR model to be used, as specified 
      in the Hugging Face model hub. For example:
      - "microsoft/trocr-small-printed" for the small model.
      - "microsoft/trocr-base-printed" for the base model.
    BEST_MODEL:
      The file path to the saved checkpoint of the best-performing model, 
      typically used for inference or further fine-tuning.
  """
  MODEL_NAME: str = "microsoft/trocr-small-printed"
  # MODEL_NAME: str = "microsoft/trocr-base-printed"
  BEST_MODEL: str = BasePathConfig.BEST_MODEL
  # BEST_MODEL: str = "microsoft/trocr-small-printed"
  # BEST_MODEL: str = "microsoft/trocr-base-printed"


@dataclass(frozen=True)
class PreprocessConfig:
  """
  A class for managing the configuration settings for data preprocessing.

  This class defines the parameters required for splitting and organizing 
  datasets for training, validation, and testing in the OCR pipeline. It 
  also references file paths for storing the processed datasets.

  Attributes:
    paths:
      An instance of `BasePathConfig` that provides the directory structure 
      and file paths for saving the processed datasets.
    TRAIN_RATIO:
      The proportion of the dataset allocated for training. 
      Default is 0.7 (70% of the data).
    VALID_RATIO:
      The proportion of the dataset allocated for validation. 
      Default is 0.15 (15% of the data).
    TEST_RATIO:
      The proportion of the dataset allocated for testing. This is computed 
      as the remainder of the dataset after training and validation allocations.
  """
  paths: BasePathConfig = BasePathConfig()
  TRAIN_RATIO: float = 0.7
  VALID_RATIO: float = 0.15
  TEST_RATIO: float = 1 - (TRAIN_RATIO + VALID_RATIO)


@dataclass(frozen=True)
class TrainingConfig:
  """A class for managing the configuration settings for model training.

  This class defines the parameters and settings used during the training 
  process, including batch size, learning rate, number of epochs, and early 
  stopping criteria. It also references file paths for saving training-related 
  outputs.

  Attributes:
    paths:
      An instance of `BasePathConfig` that provides the directory structure 
      and file paths for saving training checkpoints and results.
    BATCH_SIZE:
      The number of samples processed in each training batch.
    EPOCHS:
      The maximum number of training epochs.
    LEARNING_RATE:
      The step size for the optimizer during training. Default is 0.00005.
    EARLY_STOPPING_PATIENCE:
      The number of consecutive epochs with no improvement in validation 
      performance before training stops early. Default is 15 epochs.
    EARLY_STOPPING_THRESHOLD:
      The minimum improvement in validation loss required to reset the 
      early stopping counter. Default is 0.0001.
    SAVE_CKPT_LIMIT:
      The maximum number of model checkpoints to retain during training. 
      Once the limit is reached, older checkpoints are deleted to conserve 
      storage and maintain only the most recent ones. This helps manage disk 
      usage while ensuring access to the latest checkpoints for resuming 
      training or evaluation. It is recommended that `SAVE_CKPT_LIMIT` be 
      greater than `EARLY_STOPPING_PATIENCE` to ensure sufficient checkpoints 
      are retained for effective recovery and evaluation during 
      the early stopping process.
  """
  paths: BasePathConfig = BasePathConfig()
  BATCH_SIZE: int = 85  # 5
  EPOCHS: int = 35  # 3
  LEARNING_RATE: float = 0.00005
  EARLY_STOPPING_PATIENCE: int = 10
  EARLY_STOPPING_THRESHOLD: float = 0.0001
  SAVE_CKPT_LIMIT: int = 15


@dataclass(frozen=True)
class TestConfig:
  """A class for managing the configuration settings for model testing.

  This class defines parameters and settings related to testing the model's 
  performance, including paths for test data and results.

  Attributes:
    paths:
      An instance of `BasePathConfig` providing the directory structure 
      and file paths for loading test datasets and saving testing results.
    ACC_REPORT_TITLE:
      The title used for the accuracy report visualization, such as charts 
      or plots.
  """
  paths: BasePathConfig = BasePathConfig()
  ACC_REPORT_TITLE: str = "Fine-Tuned TrOCR-small-printed"
