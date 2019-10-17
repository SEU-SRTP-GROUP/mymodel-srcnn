#PATH FOR PROJECT   一些工程用到的路径
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  )                                                  #PROJECT ROOT DIR 项目根目录
DATASET_DIR = os.path.join(BASE_DIR,'datasets')                           #DATASET DIR 数据集目录
KITTI_DIR = os.path.join(DATASET_DIR,'KITTI')                             #KITTI DATASET DIR  kitti数据集的目录
MODEL_DIR = os.path.join(BASE_DIR,'srcnn')                                #MODEL DIR   模型的目录
LOG_DIR = os.path.join(BASE_DIR,'logs')
COCOPATH= os.path.join(BASE_DIR,'srcnn_coco.h5')