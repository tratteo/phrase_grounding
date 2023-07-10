#!/usr/bin/env python
# coding: utf-8

# # Imports
#
# Remember to run this!
#
# This is required to import the required packages!

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch
import torch.nn as nn
import cv2
import pickle
import os
import csv
import json
from PIL import Image
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import clip
from clip.model import AttentionPool2d
from clip.model import ModifiedResNet
from tqdm.auto import tqdm
from typing import Tuple, Union
from clip.model import CLIP
from torchvision.ops import generalized_box_iou_loss
from torchvision.ops.boxes import box_convert
import random
import torchvision.transforms as transforms


device = "cuda" if torch.cuda.is_available() else "cpu"


# # Dataset
#
# In this section, the **RefCOCOg** dataset is loaded and the tree splits are generated:
# - `training_data`: split used to train the model, length: **80512**
# - `validation_data`: split used to validate the model during training, length: **4896**
# - `test_data`: split used to test the model after training, length: **9602**
#
# Total: **95010**
#
# The splits include exactly all samples from the dataset. This generates a very big training dataset that, due to computational limits, cannot be effectively exploited to fine tune the model to perfection.
# Notice that the images in the dataset are around **25000**. The dataset size however is this big since each prompt, associated to the same image has been treated as a separated sample.

# **The path of the dataset must be adjusted based on the location of the dataset folder!**

# In[ ]:


# adjust based on the location of the dataset folder!
refcocog_path = "E:/DL_Datasets/refcocog"


# Load the pickle file and the instances json file

# In[ ]:


pick = pickle.load(open(refcocog_path + "/annotations/refs(umd).p", "rb"))
jsn = json.load(open(refcocog_path + "/annotations/instances.json", "rb"))
# set of all images
images_set = {}
for i in jsn["images"]:
    image_id = i["id"]
    images_set[image_id] = i

# set of all annotations
annotations_set = {}
for a in jsn["annotations"]:
    annotation_id = a["id"]
    annotations_set[annotation_id] = a

# set of all categories
categories_set = {}
remapper_cat = {}
max_cat = 0
for i, c in enumerate(jsn["categories"]):
    remapper_cat[c["id"]] = i
    c["id"] = i
    categories_set[c["id"]] = c
    if c["id"] > max_cat:
        max_cat = c["id"]
print(remapper_cat)


# **Build dataset splits**

# In[ ]:


train_data, train_label = [], []
validate_data, validate_label = [], []
test_data, test_label = [], []
for p in pick:
    data_image_path = f"{refcocog_path}/images/{images_set[p['image_id']]['file_name']}"
    data_sentences = p["sentences"]
    data_bbox = annotations_set[p["ann_id"]]["bbox"]

    data = []

    for s in data_sentences:
        sentence = s["sent"]
        data.append([data_image_path, sentence, data_bbox, p["category_id"]])

    if p["split"] == "train":
        train_data.extend(data)
    elif p["split"] == "test":
        test_data.extend(data)
    elif p["split"] == "val":
        validate_data.extend(data)

print(
    f"train {len(train_data)}, validation {len(validate_data)}, test {len(test_data)}"
)


# ### Dataset utils methods

# In[ ]:


def draw_box_on_image(image, size, bbox, color):
    w, h = size
    p1 = (int(bbox[0] * w), int(bbox[1] * h))
    p2 = (int((bbox[0] + bbox[2]) * w), int((bbox[1] + bbox[3]) * h))
    cv2.rectangle(image, p1, p2, color, 3)


def compute_target_heatmap(image, box):
    img_w, img_h = image.size
    x1 = int((box[0]) / img_w * 224)
    y1 = int((box[1]) / img_h * 224)
    x2 = int((box[0] + box[2]) / img_w * 224)
    y2 = int((box[1] + box[3]) / img_h * 224)

    target = torch.zeros((224, 224))
    target[y1 : y2 + 1, x1 : x2 + 1] = 1
    return target


def get_batch_data(batch, image_augment=False, augment_p=0.25):
    images, target_boxes, prompts, target_heatmaps, categories = [], [], [], [], []
    for image_path, prompt, box, cat in batch:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        if image_augment and random.random() < augment_p:
            augment_transform = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=(0.1, 0.6),
                        contrast=(0.4, 1),
                        saturation=(0, 0.4),
                        hue=(-0.5, 0.5),
                    )
                ]
            )
            image = augment_transform(image)

        correct_box = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
        target_boxes.append(correct_box)
        categories.append(cat)
        images.append(image)
        prompts.append(prompt)
        target_heatmaps.append(compute_target_heatmap(image, box))
    target_boxes = torch.tensor(target_boxes).to(device)
    target_boxes.requires_grad = False
    target_heatmaps = torch.stack(target_heatmaps).to(device)
    target_heatmaps.requires_grad = False
    return images, prompts, target_boxes, target_heatmaps, categories


def view_image_with_bbox(image_path, prompt, bbox):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    image = np.asarray(image)

    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    print(bbox)
    print(f"normalized: {[bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]}")
    print(f"p1 {p1}, p2 {p2}")
    cv2.rectangle(image, p1, p2, (0, 255, 255), 3)

    plt.imshow(image)
    plt.title(prompt)
    plt.show()


# # Using transfer learning: FastRCNN
#
# The pipeline is to feed the image to Fast-RCNN that returns a set of bounding boxes.
#
# Each subimage corresponding to each bounding box is extracted and then the similarity between the CLIP's encoding of the subimage and the prompt is computed. When Fast-RCNN does not return a bounding box, we use the entire image as the bounding box.
#
# The bounding box with the highest similarity is returned as output.
#
# ![fast_arch.png](attachment:fast_arch.png)
#
# - **GIoU**: `0.6`

# ### Dataset class
#
# This dataset class is created to adapt the RefCOCOg data to Fast-RCNN

# In[ ]:


class RefCOCODataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, pick, jsn, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        # load the annotations file, it also contain information of image names
        # load annotations
        self.pick = pick
        self.jsn = jsn

        set = {}
        for a in jsn["annotations"]:
            if set.get(a["image_id"]) is None:
                set[a["image_id"]] = {}
            bbox = a["bbox"]
            label = categories_set[remapper_cat[a["category_id"]]]["id"]

            if set.get(a["image_id"]).get("bboxes") is None:
                set[a["image_id"]]["bboxes"] = []
            if set.get(a["image_id"]).get("labels") is None:
                set[a["image_id"]]["labels"] = []
            b = box_convert(torch.tensor(bbox), in_fmt="xywh", out_fmt="xyxy")
            set[a["image_id"]]["bboxes"].append(b.tolist())
            set[a["image_id"]]["labels"].append(label)

        for i in jsn["images"]:
            image_id = i["id"]
            set[image_id]["file_name"] = i["file_name"]
            set[image_id]["image_id"] = image_id
        self.elems = list(set.values())
        self.tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        cur = self.elems[idx]
        # get the image path from the annoations data
        img = Image.open(self.data_dir + "/images/" + cur["file_name"]).convert("RGB")
        img_res = self.tensor(img).to(device)

        boxes = cur["bboxes"]
        num_objs = len(boxes)
        labels = cur["labels"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).to(device)
        labels = torch.as_tensor(labels).to(device)

        image_id = torch.tensor([cur["image_id"]]).to(device)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]).to(device)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs), dtype=torch.int64).to(device)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img_res, target = self.transforms(img_res, target)

        return img_res, target

    def __len__(self):
        return len(self.elems)


# In[ ]:


def get_batch(dataset, start, size):
    images = []
    targets = []
    for i in range(start, start + size):
        images.append(dataset[i][0])
        targets.append(dataset[i][1])
    return images, targets


# In[ ]:


# load the dataset
dataset = RefCOCODataset(refcocog_path, pick, jsn)


# ### Finetuning

# In[15]:


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# change head to finetune
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 80)

model.to(device)


# In[ ]:


new_params = []
pre_params = []
for p in model.named_parameters():
    if not p[1].requires_grad:
        continue
    if "roi_heads.box_predictor" in p[0]:
        new_params.append(p[1])
    else:
        pre_params.append(p[1])

optimizer = torch.optim.SGD(
    [{"params": pre_params, "lr": 5e-3}, {"params": new_params, "lr": 1e-2}],
    momentum=0.9,
    weight_decay=0.0002,
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.25)

epochs = 10
batch_size = 6
# train for 75% of the dataset elements
size = 4800  # int(len(dataset)*0.75)

model.train()
with open("finetuning.csv", "w", newline="") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(["epoch", "loss"])
for e in range(epochs):
    losses = []
    for i in tqdm(range(0, size, batch_size)):
        optimizer.zero_grad()
        images, targets = get_batch(dataset, i, batch_size)
        pred = model(images, targets)
        l = sum(loss for loss in pred.values())
        losses.append(l.item())
        l.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
    with open("finetuning.csv", "a", newline="") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow([e, sum(losses) / len(losses)])
    print(f"epoch {e} | loss {sum(losses)/len(losses)}")

torch.save(model, "fastrcnn.pt")


# In[27]:


def predict(img_path, model, threshold=0.1):
    img = Image.open(img_path)
    w, h = img.size
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img = transform(img).to(device)
    pred = model([img])

    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])]
        for i in list(pred[0]["boxes"].detach().cpu().numpy())
    ]
    pred_score = list(pred[0]["scores"].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    if len(pred_t) <= 0:
        return [[(0, 0), (w, h)]]
    pred_t = pred_t[-1]
    pred_boxes = pred_boxes[: pred_t + 1]
    return pred_boxes


def predict_and_show(img_path, threshold=0.5, rect_th=3, text_size=1, text_th=3):
    boxes, pred_cls = predict(img_path, threshold)
    img = cv2.imread(img_path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        cv2.rectangle(
            img,
            (int(boxes[i][0][0]), int(boxes[i][0][1])),
            (int(boxes[i][1][0]), int(boxes[i][1][1])),
            (0, 255, 0),
            rect_th,
        )
        cv2.putText(
            img,
            pred_cls[i],
            (int(boxes[i][0][0]), int(boxes[i][0][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (0, 255, 0),
            thickness=text_th,
        )
        plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# In[28]:


# object_detection_api(train_data[0][0], threshold=0.8)
execute = True
# model = torch.load("fastrcnn.pt")
model.to(device)
model.eval()

data = train_data + validate_data + test_data
if execute:
    base_clip, base_clip_preprocess = clip.load("RN50")
    base_clip = base_clip.to(device)
    losses = []

    for idx, data in enumerate(tqdm(data)):
        image_path, prompt, target_box, labels = data
        image = Image.open(image_path).convert("RGB")

        image_copy = np.asarray(image)
        cropped_images = []

        boxes = predict(image_path, model)

        if len(boxes) == 0:
            target_tensor = torch.tensor(target_box).to(device)
            h, w = image.size
            ans = torch.tensor([0, 0, h, w]).to(device)
            loss = generalized_box_iou_loss(ans, target_tensor)
            losses.append(loss.item())
            continue

        for ans_box in boxes:
            (x1, y1), (x2, y2) = ans_box[:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cropped_image = image_copy[y1:y2, x1:x2]
            cropped_images.append(cropped_image)

        preprocessed_images = []
        for img_np in cropped_images:
            img = Image.fromarray(img_np)
            preprocessed_image = base_clip_preprocess(img)
            preprocessed_images.append(preprocessed_image)

        cropped_image_tensors = torch.stack(preprocessed_images).to(device)

        text_tokens = clip.tokenize(prompt).to(device)
        text_tokens.shape

        with torch.no_grad():
            image_features = base_clip.encode_image(cropped_image_tensors).float()
            text_features = base_clip.encode_text(text_tokens).float()

        # divide by norm
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(image_features, text_features.T)
        # similarity

        ans = similarity.argmax()
        (x1, y1), (x2, y2) = boxes[ans][:4]

        target_tensor = torch.tensor(target_box).to(device)
        target_tensor = box_convert(target_tensor, in_fmt="xywh", out_fmt="xyxy")

        loss = generalized_box_iou_loss(
            torch.tensor([x1, y1, x2, y2]).to(device), target_tensor
        )
        losses.append(loss.item())

    print(f"GIoU: {sum(losses) / len(losses)}")
