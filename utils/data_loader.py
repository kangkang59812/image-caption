import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import json
import pdb
import cv2


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, origin_file, split, img_tags, vocab):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        if split in {'train', 'restval'}:
            self.split = ['train', 'restval']
        if split in {'val'}:
            self.split = ['val']
        if split in {'test'}:
            self.split = ['test']

        with open(origin_file, 'r') as j:
            self.origin_file = json.load(j)

        self.images_id = [(index, self.origin_file['images'][index]['imgid'])
                          for index in range(0, len(self.origin_file['images']))
                          if self.origin_file['images'][index]['split'] in self.split]

        with open(img_tags, 'r') as j:
            self.img_tags = json.load(j)

        with open(vocab, 'r') as j:
            self.vocab = json.load(j)
        self.transform = transforms.Compose([
            # transforms.RandomCrop(224, pad_if_needed=True),
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        word2id = self.vocab['word_map']
        real_index = self.images_id[index][0]  # index in origin file

        img_id = self.origin_file['images'][real_index]['imgid']
        path = self.origin_file['images'][real_index]['filepath'] + \
            '/'+self.origin_file['images'][real_index]['filename']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            try:
                image = self.transform(image)
            except:
                print('ssssss')
                pdb.set_trace()
        # Convert caption (string) to word ids.
        tags = []
        t = list(map(str.lower, self.img_tags[str(real_index)]))
        tags = [word2id[token] for token in t]
        target = torch.zeros(len(word2id))
        target[list(map(lambda n:n-1, tags))]=1
        target = torch.Tensor(target)
        return image, target

    def __len__(self):
        return len(self.images_id)

    def image_at(self, index):
        '''
        for visualization
        根据index用真实的index，self.imgid是不连续的
        '''
        real_index = self.images_id[index][0]  # index in origin file
        im_id = self.origin_file['images'][real_index]['imgid']
        path = self.origin_file['images'][real_index]['filepath'] + \
            '/'+self.origin_file['images'][real_index]['filename']
        im = Image.open(os.path.join(self.root, path)).convert('RGB')
        image_data = self.transform(im)
        tags = []
        t = list(map(str.lower, self.img_tags[str(im_id)]))
        tags = [self.vocab['word_map'][token] for token in t]
        # target = torch.zeros(len(self.vocab['word_map']))
        # target[list(map(lambda n:n-1, tags))]=1
        # target = torch.Tensor(target)
        # im = cv2.imread(os.path.join(self.root, path))
        return im, image_data, t


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    lengths = [sum(cap).item() for cap in captions]
    return images, targets, lengths


def get_loader(root, origin_file, split, img_tags, vocab, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       origin_file=origin_file,
                       split=split,
                       img_tags=img_tags,
                       vocab=vocab)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == "__main__":
    root = '/home/lkk/datasets/coco2014'
    origin_file = root+'/'+'dataset_coco.json'
    img_tags = './img_tags.json'
    voc = './vocab.json'
    # d = get_loader(root=root, origin_file=origin_file, split='train',
    #                img_tags=img_tags, vocab=voc, batch_size=1, shuffle=True, num_workers=0)
    c = CocoDataset(root=root,
                    origin_file=origin_file,
                    split='train',
                    img_tags=img_tags,
                    vocab=voc)
    im = c.image_at(0)
    for i, (imgs, tars, lens) in enumerate(d):
        images = imgs
        targets = tars
        lengths = lens
        print(images.shape, targets.shape)
        if i == 2:
            break
