# -----------------------------
# -*- coding:utf-8 -*-
# author:kangkang
# datetime:2019/4/27 12:52
# -----------------------------

from utils.utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    # coco2014 and coco2017
    create_input_files(dataset='coco2014',
                       karpathy_json_path='../../../datasets/coco2014/dataset_coco.json',
                       image_folder='../../../datasets/coco2014/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='../../../datasets/coco2014/',
                       max_len=50)
