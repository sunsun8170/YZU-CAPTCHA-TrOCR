"""Script for generating a dataset by downloading CAPTCHA images from a 
specified URL.

This script automates the collection of CAPTCHA samples, which can be used 
for training and testing machine learning models.
"""
from datetime import datetime
import getpass
import io
import os
import sys
import time
from typing import Dict, Tuple
import warnings

from bs4 import BeautifulSoup
from PIL import Image
import requests
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, logging
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from src.configs import BotConfig, ModelConfig


class Bot:
  """A bot for automatically logging into the YZU course selection system and 
  downloading CAPTCHA images.

  Attributes:
    _account:
      The user account used for login.
    _password:
      The user password used for login.
    _time_now:
      The current time in string format.
    _device:
      The device used for model inference, either "cuda" for GPU or "cpu" for CPU.
    _processor:
      The TrOCR processor used for CAPTCHA recognition.
    _model:
      The Vision Encoder and Text Decoder model used for OCR tasks.
    _session:
      The HTTP session object used for making requests to the YZU course selection system.
  """

  def __init__(self) -> None:
    """
    Initialize the bot with the provided configuration parameters.

    Args:
      config:
        A configuration object that contains the necessary parameters 
        to set up the bot.
    """
    self._account = ""
    self._password = ""

    self._time_now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    self._device = "cuda" if torch.cuda.is_available() else "cpu"
    self._processor, self._model = self._init_model()
    self._model.generation_config.max_new_tokens = 4

    self._session = self._init_session()

  def _init_model(self) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load the TrOCR processor and the vision encoder-decoder model.

    Returns:
      tuple:
        A tuple containing two elements:
          - The TrOCR processor.
          - The TrOCR vision encoder-decoder model for OCR tasks.

    Raises:
      Exception: 
        If the model fails to load or there is an error in loading the processor
        or model.
    """
    try:
      processor = TrOCRProcessor.from_pretrained(
          ModelConfig.MODEL_NAME,
          clean_up_tokenization_spaces=True,
          char_level=True,
      )
      model = VisionEncoderDecoderModel.from_pretrained(
          ModelConfig.MODEL_NAME).to(self._device)
      return processor, model

    except Exception as e:
      print(f"{self._time_now}   [ 模型載入失敗 ]\n詳細資訊: {e}")
      sys.exit(0)

  def _init_session(self) -> requests.Session:
    """Initialize and return a new requests session.

    Returns:
      requests.Session:
        A new session object for making HTTP requests, allowing for connections 
        and automatic handling of cookies.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent":
            "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
        "Accept":
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8",
        "Accept-Language":
            "en-US,en;q=0.5",
        "Referer":
            BotConfig.LOGIN_URL_REFERER,
        "Accept-Encoding":
            "gzip, deflate, br, zstd",
        "Upgrade-Insecure-Requests":
            "1",
    })

    return session

  def _init_login_payload(self, **kwargs) -> Dict[str, str]:
    """Prepare and initialize the login payload with the required parameters.

    Args:
      kwargs:
        Optional key-value pairs to include in the payload, such as user-specific 
        parameters or additional fields required for the login process.

    Returns:
      dict:
        A dictionary representing the login payload, containing all necessary 
        values for the authentication request.
    """
    payload = {
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "Txt_User": self._account,
        "Txt_Password": self._password,
        "btnOK": "確定"
    }
    payload.update(kwargs)

    return payload

  def _get_captcha_text(self, image: Image.Image) -> str:
    """Perform Optical Character Recognition (OCR) on the provided CAPTCHA image.

    Args:
      image:
        A PIL image object containing the CAPTCHA to be recognized.

    Returns:
      str:
        The recognized text from the CAPTCHA, expected to be exactly 4 characters long.

    Raises:
      Exception:
        If the OCR model fails to process the image or recognize the text.
    """
    try:
      pixel_values = self._processor(
          image,
          return_tensors="pt",
      ).pixel_values.to(self._device)
      generated_ids = self._model.generate(pixel_values)
      return self._processor.batch_decode(
          generated_ids,
          skip_special_tokens=True,
      )[0]
    except Exception as e:
      print(f"{self._time_now}   [ 模型辨識錯誤 ] 在辨識驗證碼文字時發生了未知的錯誤!\n詳細資訊: {e}")
      sys.exit(0)

  def _login(self) -> bool:
    """Log in to the system using the account credentials and CAPTCHA.

    If the login is successful (indicating the CAPTCHA text was entered correctly), 
    the CAPTCHA image will be saved for future reference or analysis.

    Returns:
      bool:
        True if the login is successful, otherwise False.
    """
    while True:
      # if the network is available
      try:
        self._session.cookies.clear()

        captcha_response = self._session.get(
            BotConfig.CAPTCHA_URL,
            stream=True,
            timeout=BotConfig.REQUEST_TIMEOUT,
        )
        captcha_response.raise_for_status()

        captcha_data = io.BytesIO(captcha_response.content)
        captcha_img = Image.open(captcha_data).convert("RGB")
        captcha_text = self._get_captcha_text(captcha_img)

        login_response = self._session.get(BotConfig.LOGIN_URL)
        login_response.raise_for_status()

        if "選課系統尚未開放" in login_response.text:
          print(f"{self._time_now}   [ 選課系統尚未開放 ]")
          sys.exit(0)

        parser = BeautifulSoup(login_response.text, "lxml")

        login_payload = self._init_login_payload(
            __VIEWSTATE=parser.select_one("#__VIEWSTATE")["value"],
            __VIEWSTATEGENERATOR=parser.select_one("#__VIEWSTATEGENERATOR")
            ["value"],
            __EVENTVALIDATION=parser.select_one("#__EVENTVALIDATION")["value"],
            DPL_SelCosType=parser.select_one(
                "#DPL_SelCosType option:not([value='00'])")["value"],
            Txt_CheckCode=captcha_text,
        )

        result = self._session.post(
            BotConfig.LOGIN_URL,
            data=login_payload,
            timeout=BotConfig.REQUEST_TIMEOUT,
        )
        result.raise_for_status()

        # if CAPTCHA is correct
        if "parent.location ='SelCurr.aspx?Culture=zh-tw'" in result.text:
          filename = os.path.join(
              BotConfig.paths.CAPTCHAS,
              f"{captcha_text}.png",
          )

          # if CAPTCHA image doesn't exist, save it
          if not os.path.exists(filename):
            captcha_img.save(filename, "PNG")
            return True

          # if CAPTCHA image exist
          else:
            return False

        # if CAPTCHA is wrong
        elif "驗證碼錯誤" in result.text:
          return False

        # if other error occurs
        else:
          print(f"{self._time_now}   [ 錯誤 ] 帳號密碼錯誤或未在此階段選課時程之內!")
          sys.exit(0)

      # if the network is unstable or unavailable
      except requests.RequestException as e:
        print(f"{self._time_now}   [ 網路異常 ] 嘗試連線中!\n詳細資訊: {e}")
        time.sleep(0.5)

      # if other error occurs
      except Exception as e:
        print(f"{self._time_now}   [ 未知的錯誤 ]\n詳細資訊: {e}")
        sys.exit(0)

  def run(self) -> None:
    """The main entry point for executing the bot.

    This function orchestrates the bot's workflow, including logging in, 
    handling CAPTCHA recognition, and performing other automated tasks.
    """
    print("Please enter your YZU Portal account and password. "
          "Your password will be hidden while typing.")
    print("Once logged in, the bot will automatically download CAPTCHAs.")
    self._account = input("Account: ")
    self._password = getpass.getpass(prompt="Password: ")

    total = 0

    with tqdm(
        total=BotConfig.NUM_TO_DOWNLOAD,
        desc="CAPTCHAs",
        position=0,
    ) as bar:
      while total <= BotConfig.NUM_TO_DOWNLOAD:
        if self._login():
          total += 1
          bar.update(1)

    print(f"{self._time_now}   [ 完成抓取驗證碼圖形 ] "
          f"已儲存共 {BotConfig.NUM_TO_DOWNLOAD} 張驗證碼圖片!")


def main() -> None:
  """The main function for download CAPTCHA images from the YZU course 
  selection system.

  This function automates the process of accessing the system and retrieving CAPTCHA 
  images for dataset generation or analysis.
  """
  warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
  logging.set_verbosity_error()
  Bot().run()
