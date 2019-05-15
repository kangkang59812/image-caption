# -----------------------------
# -*- coding:utf-8 -*-
# author:kangkang
# datetime:2019/4/27 14:43
# -----------------------------
import json
import re
path = '../../../datasets/coco2017/annotations/'

val = json.load(open(path+'captions_val2017.json', 'r'))
train = json.load(open(path+'captions_train2017.json', 'r'))

# combine all images and annotations together
imgs = val['images'] + train['images']
annots = val['annotations'] + train['annotations']

dataset = {'dataset':'coco2017'}
itoa = {}
for a in annots:
    imgid = a['image_id']
    if not imgid in itoa: itoa[imgid] = []
    itoa[imgid].append(a)
out=[]
for i,img in enumerate(imgs):
    #if i<=100:
    item = {}
    item["filepath"]= "train2017" if 'train' in img['coco_url'] else "val2017"
    item["sentids"] = [s_id['id'] for s_id in itoa[img['id']]]
    item["filename"] = img["file_name"]
    item["imgid"] = i
    item["split"] = item["filepath"][:-4]
    item["sentences"] = []
    for cap in itoa[img['id']]:
        s_item={}
        s_item["tokens"]=re.sub("[^a-zA-Z\s]",'',cap["caption"]).split()
        s_item["raw"]=cap["caption"]
        s_item["imgid"]=item["imgid"]
        s_item["sentid"]=cap["id"]
        item["sentences"].append(s_item)
    item["cocoid"]=img["file_name"][:-4].lstrip('0')
    out.append(item)
result={"images":out}
result.update(dataset)

json.dump(result,open('dataset_coco2017.json','w'))