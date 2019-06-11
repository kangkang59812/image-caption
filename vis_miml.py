
from utils.utils import plot_instance_attention, plot_instance_probs_heatmap
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from utils.data_loader import CocoDataset
from models.miml import MIML
import json
import torch

features = torch.empty(1, 196, 1024).cuda()
instance_probs = None


def hook(module, input, ouput):
    global features, instance_probs
    features = torch.empty(1, 196, 1024).cuda()
    features.copy_(ouput.data)
    features = features.permute(0, 2, 1).reshape(-1, 1024, 1, 196)
    instance_probs = features.permute(0, 3, 1, 2)[:, :, :, 0].squeeze().cpu()
    # print("instance_probs.shape=", instance_probs.shape)

    # plot instance label score
    plot_instance_probs_heatmap(instance_probs, './1.jpg')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str,
                        default="/home/lkk/code/caption_v1/checkpoint/MIML.pth.tar")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    miml_model_path = args.model
    root = '/home/lkk/datasets/coco2014'
    origin_file = root+'/'+'dataset_coco.json'
    img_tags = './img_tags.json'
    voc = './vocab.json'
    dataset = CocoDataset(root=root,
                          origin_file=origin_file,
                          split='val',
                          img_tags=img_tags,
                          vocab=voc)
    choose = np.random.randint(0, len(dataset), 10)
    with open(voc, 'r') as j:
        vocab = json.load(j)

    cls_names = vocab['map_word']
    model = MIML()
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    checkpoint = torch.load(
        '/home/lkk/code/caption_v1/checkpoint/MIML.pth.tar')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    for it in choose:
        im, image_data, target = dataset.image_at(it)

        # heat map
        handle = model.module.sub_concept_layer.softmax1.register_forward_hook(
            hook)
        label_id_list = np.where(model(image_data.unsqueeze(
            0).cuda()).cpu().detach().numpy() > 0.5)[0]
        handle.remove()
        label_name_list = [cls_names[str(i+1)] for i in label_id_list]
        instance_points, instance_labels = [], []
        for _i, label_id in enumerate(label_id_list):
            max_instance_id = np.argmax(instance_probs[:, label_id])
            conv_y, conv_x = max_instance_id / 14, max_instance_id % 14
            instance_points.append(((conv_x * 16 + 8), (conv_y * 16 + 8)))
            instance_labels.append(label_name_list[_i])
        im_plot = cv2.resize(np.array(im), (224, 224)).astype(
            np.uint8)[:, :, (0, 1, 2)]
        plot_instance_attention(im_plot, instance_points,
                                instance_labels, save_path='./vis_4/'+str(it)+'.jpg')
        print(target)
        print(instance_labels)
        print('****************')
