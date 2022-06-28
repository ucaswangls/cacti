import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch 
from cacti.utils.mask import generate_masks
from cacti.utils.utils import get_device_info
from cacti.utils.metrics import compare_psnr,compare_ssim
from cacti.utils.config import Config
from cacti.models.builder import build_model
from cacti.utils.logger import Logger

if __name__=="__main__":
    print("Welcome to the Video SCI repository!")