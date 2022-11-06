#@title **セットアップ**

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#os.chdir('../')
CODE_DIR = 'VToonify'
device = 'cuda'

!git clone https://github.com/cedro3/VToonify.git $CODE_DIR

!wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
!sudo unzip ninja-linux.zip -d /usr/local/bin/
!sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
!pip install wget

os.chdir(f'./{CODE_DIR}')
MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()), CODE_DIR, 'checkpoint')
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), CODE_DIR, 'data')
OUT_DIR = os.path.join(os.path.dirname(os.getcwd()), CODE_DIR, 'output')

%load_ext autoreload
%autoreload 2
import sys
sys.path.append(".")
sys.path.append("..")
import argparse
import numpy as np
import cv2
import dlib
import torch
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.vtoonify import VToonify
from model.bisenet.model import BiSeNet
from model.encoder.align_all_parallel import align_face
from util import save_image, load_image, visualize, load_psp_standalone, get_video_crop_parameter, tensor2cv2

MODEL_PATHS = {
    "encoder": {"id": "1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej", "name": "encoder.pt"},
    "faceparsing": {"id": "1jY0mTjVB8njDh6e0LP_2UxuRK3MnjoIR", "name": "faceparsing.pth"},
    "arcane_exstyle": {"id": "1TC67wRJkdmNRZTqYMUEFkrhWRKKZW40c", "name": "exstyle_code.npy"},
    "caricature_exstyle": {"id": "1xr9sx_WmRYJ4qHGTtdVQCSxSo4HP3-ip", "name": "exstyle_code.npy"},
    "cartoon_exstyle": {"id": "1BuCeLk3ASZcoHlbfT28qNru4r5f-hErr", "name": "exstyle_code.npy"},
    "pixar_exstyle": {"id": "1yTaKuSrL7I0i0RYEEK5XD6GI-y5iNUbj", "name": "exstyle_code.npy"},
    "arcane000": {"id": "1pF4fJ8acmawMsjjXo4HXRIOXeZR8jLVh", "name": "generator.pt"},
    "arcane077": {"id": "16rLTF2oC0ZeurnM6hjrfrc8BxtW8P8Qf", "name": "generator.pt"},
    "caricature039": {"id": "1C1E4WEoDWzl0nAxR9okKffFmlMOENbeF", "name": "generator.pt"},
    "caricature068": {"id": "1B1ko1x8fX2aJ4BYCL12AnknVAi3qQc8W", "name": "generator.pt"},
    "cartoon026": {"id": "1YJYODh_vEyUrL0q02okjcicpJhdYY8An", "name": "generator.pt"},
    "cartoon299": {"id": "101qMUMfcI2qDxEbfCBt5mOg2aSqdTaIt", "name": "generator.pt"},
    "pixar052": {"id": "16j_l1x0DD0PjwO8YdplAk69sh3-v95rr", "name": "generator.pt"},
    "cartoon": {"id": "11s0hwhZWTLacMAzZH4OU-o3Qkp54h30J", "name": "generator.pt"},
}


# download all checkpoint
import gdown
gdown.download('https://drive.google.com/uc?id=1l8aqqZJMLnmQtkz8HPzakblq5G_zMZAr', 'v1.zip', quiet=False)
! unzip -d checkpoint v1.zip
gdown.download('https://drive.google.com/uc?id=1hBcGhPmhZi01Jg6mvwD1yPyMVJgeiExX', 'v2.zip', quiet=False)
! unzip -d checkpoint v2.zip
gdown.download('https://drive.google.com/uc?id=11aTPKbdnqjaljyzhMCVLz07Q3-74Bgsj', 'v3.zip', quiet=False)
! unzip -d checkpoint v3.zip

%matplotlib inline
from function import *
! mkdir download