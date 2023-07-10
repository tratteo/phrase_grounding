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


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
# 
# The training process has not been carried out on all the train data, due to computational and time limits.

# **The path of the dataset must be adjusted based on the location of the dataset folder!**

# In[ ]:


# adjust based on the location of the dataset folder!
refcocog_path =  "E:/DL_Datasets/refcocog"


# Load the pickle file and the instances json file

# In[ ]:


pick = pickle.load(open(refcocog_path+"/annotations/refs(umd).p", "rb"))
jsn = json.load(open(refcocog_path+"/annotations/instances.json", "rb"))


# In[ ]:


# set of all images
images_set = {}
for i in jsn['images']:
  image_id = i['id']
  images_set[image_id] = i

# set of all annotations
annotations_set = {}
for a in jsn['annotations']:
  annotation_id = a['id']
  annotations_set[annotation_id] = a

# set of all categories
categories_set = {}
max_cat=0
for c in jsn['categories']:
  category_id = c['id']
  categories_set[category_id] = c
  if c['id'] > max_cat:
    max_cat = c['id']


# **Build dataset splits**

# In[ ]:


train_data, train_label       = [], []
validate_data, validate_label = [], []
test_data, test_label         = [], []
for p in pick:
    data_image_path = f"{refcocog_path}/images/{images_set[p['image_id']]['file_name']}"
    data_sentences = p['sentences']
    data_bbox = annotations_set[p['ann_id']]['bbox']
    
    data = []

    for s in data_sentences:
        sentence = s['sent']
        data.append([data_image_path, sentence, data_bbox, p["category_id"]])

    if p['split'] == 'train':
        train_data.extend(data)
    elif p['split'] == 'test':
        test_data.extend(data)
    elif p['split'] == 'val':
        validate_data.extend(data)

print(f"train {len(train_data)}, validation {len(validate_data)}, test {len(test_data)}")


# ### Dataset utils methods

# In[ ]:


def draw_box_on_image(image,size, bbox, color):
    w, h = size
    p1 = (int(bbox[0]*w), int(bbox[1]*h))
    p2 = (int((bbox[0]+bbox[2])*w), int((bbox[1] + bbox[3])*h))
    cv2.rectangle(image, p1, p2, color, 3)

def compute_target_heatmap(image, box):
    img_w, img_h = image.size
    x1 = int((box[0]) / img_w * 224)
    y1 = int((box[1]) / img_h * 224)
    x2 = int((box[0] + box[2]) / img_w * 224)
    y2 = int((box[1] + box[3]) / img_h * 224)

    target = torch.zeros((224, 224))
    target[y1:y2+1, x1:x2+1] = 1
    return target

def get_batch_data(batch, image_augment=False, augment_p=0.25):
    images, target_boxes, prompts, target_heatmaps, categories = [], [], [], [], []
    for image_path, prompt, box, cat in batch:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        if image_augment and random.random()<augment_p:
            augment_transform = transforms.Compose([
                transforms.ColorJitter(brightness=(0.1,0.6), contrast=(0.4, 1),saturation=(0, 0.4), hue=(-0.5, 0.5))
            ])
            image = augment_transform(image)
            
        correct_box = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
        target_boxes.append(correct_box)
        categories.append(cat)
        images.append(image)            
        prompts.append(prompt)        
        target_heatmaps.append(compute_target_heatmap(image, box))    
    target_boxes = torch.tensor(target_boxes).to(device)
    target_boxes.requires_grad=False
    target_heatmaps = torch.stack(target_heatmaps).to(device)
    target_heatmaps.requires_grad=False
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
    cv2.rectangle(image, p1, p2, (0,255,255), 3)

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
# - **GIoU**: `0.6`

# ### Dataset class
# 
# This dataset class is created to adapt the RefCOCOg data to Fast-RCNN

# In[ ]:


finetuning_data = {}

class RefCOCODataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        # load the annotations file, it also contain information of image names
        # load annotations
        self.pick = pickle.load(open(self.data_dir+ "/annotations/refs(umd).p", "rb"))
        self.jsn = json.load(open(self.data_dir+ "/annotations/instances.json", "rb"))

        set = {}
        for a in jsn['annotations']:
            if set.get(a["image_id"]) is None:
                set[a["image_id"]] = {}
            bbox = a["bbox"]
            label = categories_set[a["category_id"]]["id"]
            
            if set.get(a["image_id"]).get("bboxes") is None:
                set[a["image_id"]]["bboxes"] = []
            if set.get(a["image_id"]).get("labels") is None:
                set[a["image_id"]]["labels"] = []
            b = box_convert(torch.tensor(bbox), in_fmt="xywh",out_fmt="xyxy")
            set[a["image_id"]]["bboxes"].append(b.tolist())    
            set[a["image_id"]]["labels"].append( label)   

        for i in jsn['images']:
            image_id = i['id']
            set[image_id]["file_name"] = i["file_name"]     
            set[image_id]["image_id"] = image_id
        self.elems = list(set.values())
        self.tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):

        cur = self.elems[idx]
        # get the image path from the annoations data
        img = Image.open(self.data_dir +"/images/"+cur["file_name"]).convert("RGB")
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

        return  img_res, target

    def __len__(self):
        return len(self.elems)


# In[ ]:


def get_batch(dataset, start, size):
    images = []
    targets = []
    for i in range(start, start+size):
        images.append(dataset[i][0])
        targets.append(dataset[i][1])
    return images, targets

# load the dataset
dataset = RefCOCODataset(refcocog_path)


# ### Finetuning

# In[ ]:


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# cange head to finetune
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, max_cat) 

model.to(device)


# In[ ]:


new_params = []
pre_params = []
for p in model.named_parameters():
    if not p[1].requires_grad: continue
    if "roi_heads.box_predictor" in p[0]:
        new_params.append(p[1])
    else:
        pre_params.append(p[1])

optimizer = torch.optim.SGD([
    {"params":pre_params, "lr":5E-3},
    {"params":new_params, "lr":1E-2}
        ], momentum=0.9, weight_decay=0.0002)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=4, gamma=0.25)

epochs = 64
batch_size = 8
# train for 75% of the dataset elements
size = int(len(dataset)*0.75)

model.train()
for e in range(epochs):
    losses = []
    for i in range(0, 8, batch_size):
        images, targets = get_batch(dataset, i, batch_size)
        pred = model(images, targets)
        l = sum(loss for loss in pred.values())
        losses.append(l.item())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
    print(f"epoch {e} | loss {sum(losses)/len(losses)}")
    
torch.save(model, "fastrcnn.pt")


# In[ ]:


def predict(img_path, threshold=0.8):
  img = Image.open(img_path)
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  img = transform(img).to(device)
  pred = model([img])

  pred_class = [categories_set[i] for i in list(pred[0]['labels'].detach().cpu().numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
  pred_score = list(pred[0]['scores'].detach().cpu().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes, pred_class

def predict_and_show(img_path, threshold=0.5, rect_th=3, text_size=1, text_th=3): 
  boxes, pred_cls = predict(img_path, threshold) 
  img = cv2.imread(img_path) 
  img = np.array(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
  for i in range(len(boxes)): 
    cv2.rectangle(img, (int(boxes[i][0][0]),int(boxes[i][0][1])), (int(boxes[i][1][0]),int(boxes[i][1][1])), (0, 255, 0), rect_th) 
    cv2.putText(img,pred_cls[i], (int(boxes[i][0][0]),int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) 
    plt.figure(figsize=(10,8)) 
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()


# In[ ]:


#object_detection_api(train_data[0][0], threshold=0.8)
execute = True

if execute:
    base_clip, base_clip_preprocess = clip.load("RN50")
    base_clip = base_clip.to(device)
    losses = []

    for idx, data in enumerate(tqdm(train_data + validate_data + test_data)):
        image_path, prompt, target_box = data
        image =  Image.open(image_path).convert("RGB")

        image_copy = np.asarray(image)
        cropped_images = []

        boxes, pred_cls = predict(image_path)

        if len(boxes) == 0:
            target_tensor = torch.tensor(target_box).to(device)
            h, w = image.size
            ans = torch.tensor([0, 0, h, w]).to(device)
            loss = generalized_box_iou_loss(ans, target_tensor)
            losses.append(loss.item())
            continue

        for ans_box in boxes:
            (x1, y1), (x2, y2) = ans_box[:4]
            x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
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
        #similarity

        ans = similarity.argmax()
        (x1, y1), (x2, y2) = boxes[ans][:4]

        target_tensor = torch.tensor(target_box).to(device)
        target_tensor = box_convert(target_tensor, in_fmt="xywh", out_fmt="xyxy")

        loss = generalized_box_iou_loss(torch.tensor([x1,y1,x2,y2]).to(device), target_tensor)
        losses.append(loss.item())

    print(f"GIoU: {sum(losses) / len(losses)}")


# # Model
# 
# ![architecture.png](attachment:architecture.png)
# 
# The architecture proposed, is inspired by the recent work: [Adapting CLIP For Phrase Localization Without Further Training](https://arxiv.org/abs/2204.03647) by _Jiahao Li, Greg Shakhnarovich, Raymond A. Yeh_.
# 
# In their paper, the goal was to adapt the **CLIP** model to phrase localization without the need of any further training. This goal is a perfect starting point in order to adapt their solution to out phrase grounding task for the project.
# 
# To begin with, we have taken the "backbone" model from their repository: https://github.com/pals-ttic/adapting-CLIP.
# The code borrowed includes the modification of the ResNet of CLIP in order to introduce, in the last layer, a spatial attention layer.
# The goal is to adapt the CLIP model for phrase localization and, since CLIP effectively acts as an image and text embedder, there is the need to adapt the model to introduce spatial reasoning.
# The steps are:
# - Extract spatial features **mantaining their semantic meaning** (alignment with text)
# - Compute the inner product with the text embedding, effectively generating a score map (heatmap)
# 
# -----
# 
# ##### Single stage model
# 
# Many proposed methods that attempts to perform transfer learning to phrase localization and grounding starting from famous conv networks such as Fast-RCNN or Mask-RCNN, implements a what is called **two step models** [2].
# 
# Two stage models entails the use of an external feature extractor such as a Convolutional neural network that usually performs object detection and is able to extract bounding boxes.
# These candidates boxes are then fed to the CLIP image encoder and compared with the encoding of the prompt, outputting the box with the highest score.
# These solutions have an important caveat: since the CLIP encoding is compared with the subimage composed of the bounding box only, spatial reasoning is not included into the model.
# 
# The proposed method can be instead considered a **one stage model**.
# Apart from the performance that we were able to achieve, the model should in theory be able to convey the spatial reasoning into the box regressor.
# As shown below, the CLIP model is slightly modified in order to include this feature. This is the reason why the model is able to infer the heatmaps with quite remarkable accuracy.
# 
# 
# It is worth noticing that, although the heatmaps are extracted with the `ResNetHighResV2` model and fed into the `HeatmapToBox` regressor, the framework can be considered as a single stage. 
# Both the models could be infact unified in order to provide a single interface. This has not been done in order to better delineate the original contribution of the paper and our contribution.

# ### Custom Spatial CLIP
# 
# In this cells, the code inspired from **Adapting CLIP For Phrase Localization Without Further Training** is introduced.
# It includes the customized `AttentionSpatial2d` which basically introduces spatial attention.
# 
# The `ModifiedSpatialResNet` is a simple ResNet which includes the previously introduced `AttentionSpatial2d`.
# 
# Finally, the custom CLIP with the `ModifiedSpatialResNet` is defined.
# 
# The final method `build_feature_extractor_model` has the job of creating this customized CLIP and to enable **transfer learning** from CLIP by copying the relative weights into the custom model.

# In[ ]:


clip_model, clip_preprocess = clip.load("RN50",jit=False,device=device)
clip_model = clip_model.to(device)

def linear(x, weight, bias):
    x = x.matmul(weight.t())
    x += bias
    return x

class AttentionSpatial2d(AttentionPool2d):
    """Edited attention pool layer to introduce spatial attention"""
    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__(spacial_dim, embed_dim, num_heads, output_dim)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h*w).permute(2, 0, 1) # NCHW -> (H*W)NC
        x = linear(x, self.v_proj.weight, self.v_proj.bias)
        x = linear(x, self.c_proj.weight, self.c_proj.bias)
        x = x.permute(1, 2, 0).reshape(n, -1, h, w) # (H*W)NC -> C(H*W)N -> (N, -1, H, W)
        return x

class ModifiedSpatialResNet(ModifiedResNet):
    """Modified resnet to include the edited attention pool layer"""
    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 input_resolution=224,
                 width=64):
        super().__init__(layers, output_dim, heads, input_resolution, width)

        self.attnpool = AttentionSpatial2d(
            input_resolution // 32, width * 32, heads, output_dim)

class CLIPSpatialResNet(CLIP):
    """Modified spatial CLIP including the spatial attention"""
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):

        super().__init__(embed_dim, image_resolution, vision_layers, vision_width,
                         vision_patch_size, context_length, vocab_size,
                         transformer_width, transformer_heads, transformer_layers)

        # Override the visual model
        vision_heads = vision_width * 32 // 64
        self.visual = ModifiedSpatialResNet(layers=vision_layers,
                                            output_dim=embed_dim,
                                            heads=vision_heads,
                                            input_resolution=image_resolution,
                                            width=vision_width)        


    def forward(self, image):
        image = image.type(self.dtype)

        # pad image
        pad = 64
        pad = (pad, pad, pad, pad)
        padded_image = F.pad(image, pad, "constant", 0)

        # get features
        features = self.encode_image(padded_image)
        target_size_h, target_size_w = image.size(-2) // 32, image.size(-1) // 32

        # compute new pad size
        pad_h = (features.size(-2) - target_size_w) // 2
        pad_w = (features.size(-1) - target_size_w) // 2
        features = features[:, :, pad_h:pad_h+target_size_h, pad_w:pad_w+target_size_w]

        # interpolate back to 224*224
        features = F.upsample(features, size=(image.size(-2), image.size(-1)),
            mode="bilinear", align_corners=None) # 1*C*H*W

        return features
    

def build_feature_extractor_model(clip_model): 
    """"Instantiate the modified CLIP model and adapt weights"""
    # transfer learning: extract weights from CLIP
    clip_state_dict = clip_model.state_dict()
    # run [k for k in clip_state_dict if k.startswith("visual.layer2")] to see what's up
    counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
    vision_layers = tuple(counts)
    vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
    output_width = round(
        (clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)

    vision_patch_size = None
    image_resolution = output_width * 32

    embed_dim = clip_state_dict["text_projection"].shape[1]
    context_length = clip_state_dict["positional_embedding"].shape[0]
    vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
    transformer_width = clip_state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(
        k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIPSpatialResNet(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)#.to(device)
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in clip_state_dict:
            del clip_state_dict[key]


    # False for the average filter layer.
    model.load_state_dict(clip_state_dict, strict=False)
    model.eval()
    if device == 'cpu':
        model.float()
    for param in model.parameters():
            param.requires_grad = False
    # for param in model.visual.attnpool.parameters():
    #     param.requires_grad = True
    
    #convert_weights(model)
    return model


# ### The Feature extractor: Heatmaps generator
# 
# Here, the actual feature extractor model is created.
# 
# The `ResNetHighResV2` model is responsible for taking as input images and prompts, generating the scoremaps (heatmaps).
# 
# > The model has been adapted and customized (V2) in order to enable batch computation of heatmaps, since in the original implementation of the paper, the model was able to accept single image and prompt.
# 
# -----
# Example of an image and its corresponding heatmap. Notice that the heatmaps is _shrinked_ and normalized to a fixed width and height, nonetheless spatial reasoning is mantained.
# 
# `the white imac computer that is also turned on`
# 
# 
# ![1.png](attachment:1.png) ![h_1.png](attachment:h_1.png)

# #### Generator

# In[ ]:


class ResNetHighResV2(nn.Module):
    """Feature extractor that includes CLIP as its fundation model"""
    def __init__(self, clip_preprocess, clip_model, tokenize, temperature=0.1, remap_heatmaps=True):
        super().__init__()
        self.spatial_model = build_feature_extractor_model(clip_model)
        self.clip_preprocess = clip_preprocess
        self.tokenize = tokenize
        self.temperature = temperature
        self.remap_heatmaps=remap_heatmaps        

    def get_image_features(self, images):
        images = [clip_preprocess(image) for image in images]
        images = torch.stack(images).to(device)
        image_features = self.spatial_model(images)
        return image_features

    def get_text_features(self, texts):
        tokenized_texts = self.tokenize(texts).to(device)
        text_features = self.spatial_model.encode_text(tokenized_texts)
        return text_features
    
    def get_heatmaps(self, image_features, text_features):
        heatmaps = ((image_features / image_features.norm(dim=1, keepdim=True)) * (text_features / text_features.norm(dim=1, keepdim=True))[:, :, None, None]).sum(1)
        heatmaps = torch.exp(heatmaps/self.temperature)
        if self.remap_heatmaps:
            norm_heatmaps = torch.tensor(heatmaps)
            for i in range(len(heatmaps)):
                min = torch.min(heatmaps[i])
                norm_heatmaps[i] = (heatmaps[i] - min) / (torch.max(heatmaps[i])-min) + 1E-3    
            heatmaps = norm_heatmaps            
        heatmaps = heatmaps.pow(5)
        return heatmaps

    def forward(self, images, texts):
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(texts)
        heatmaps = self.get_heatmaps(image_features, text_features)
        return heatmaps.squeeze(dim=1)


# ### Bounding Box regressor
# 
# The following model is the **head** of all of our framework.
# 
# It takes as input the heatmaps generated by `ResNetHighResV2` and remapped by the `HeatmapRemapper` and regresses the four points of the bounding box in the form `(x, y, w, h)`.
# 
# It is worth noticing that this choice of a single bounding box regression without label is motivated by the fact that in the **RefCOCOg** dataset, each (image, prompt) is associated to a single bounding box.
# Hence, since the job was to generate a phrase localization framework on the **RefCOCOg** dataset, the choice was to have a regressor that simply outputs a single box.
# 
# -----
# 
# The model effectively treats the heatmap as an image.
# 
# The first sequential layer is a series of `Conv2d` layers, that have the job to comprehend the spatial structure of the heatmap.
# 
# The second sequential layer is composed of an `AvgPool1d` that has the job to smooth the heatmap, followed by a **FFNN** that effectively regresses the four points.

# In[ ]:


class HeatmapToBox(nn.Module):
    """Custom model to regress a bounding box from an heatmap"""
    def __init__(self):
        super().__init__()  
        self.seq = nn.Sequential(                            
            nn.Conv2d(1,1,9,stride=1), 
            nn.Conv2d(1,1,7,stride=2), 
            nn.Conv2d(1,1,3,stride=2),
            
            nn.Flatten(),
            nn.AvgPool1d(4),

            nn.Linear(676, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),             
            nn.Sigmoid(), 

            nn.Linear(256, 128),  
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2),          
            nn.Sigmoid(), 

            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        print(f"bboxer parameters: {self.params_count()}")
         
    def params_count(self):
        c = 0
        for p in self.parameters():
            c += p.numel()
        return c

    def forward(self, x):   
        return self.seq(x.unsqueeze(dim=1))


# # Training
# 
# The training process is divided into two:
# - Finetuning the custom **Spatial CLIP**     
#     This step aims in enforcing the spatial CLIP to generate good heatmaps. The loss is a `MSELoss` with respect to binary target heatmaps. These targets are basically heatmaps with the value 1 only on pixels inside the bounding box and 0 otherwise.
#     
#     **The pipeline code is present below, however we noticed that finetuning even only the last layer of the `ModifiedResNet` model requires huge amount of computational time, even when other layers are frozen. Therefore extensive training is not been carried out.**
#     
# - Training the **Box Regressor**     
#     In this step we train our custom model to regress the actual bounding box
# 

# **Intersection Over Union**
# 
# The following function from `torchvision` calculates a gradient friendly version of the [Generalized Intersection Over Union](https://giou.stanford.edu/) and treats it as a loss, meaning that the `torch.Tensor` result is reduced to a single float that represents the average value for all the bounding boxes.
# 
# Boxes are converted from the **RefCOCOg** format to the format required by `torchvision`.

# In[ ]:


def iou(boxes1, boxes2) -> torch.Tensor:
    return generalized_box_iou_loss(box_convert(boxes1,in_fmt="xywh",out_fmt="xyxy"),box_convert(boxes2,in_fmt="xywh",out_fmt="xyxy"),reduction="mean")


# **Training parameters**

# In[ ]:


train_size = 128
train_batch_size = 8
epochs = 64
mini_train_data = train_data[:train_size]

validation_size = 512
validation_batch_size = 16
mini_val_data = validate_data[:validation_size]

test_size = 256
test_batch_size = 16
mini_test_data = test_data[:test_size]


# #### Utility functions

# In[ ]:


import os

def evaluate_batch_routine(model, loss_fn, feature_extractor,data_batch, graphical=False, save=False):
    get_ipython().run_line_magic('matplotlib', 'inline')
    if save:
        try: 
            os.mkdir("imgs") 
        except OSError as error: 
            pass 
    model.eval()
    images, prompts, target_boxes, target_heatmaps = get_batch_data(data_batch)    
    with torch.no_grad():
        heatmaps = feature_extractor(images, prompts)
        prediction_boxes = model(heatmaps.to(device))

    c=0
    if graphical:
        for img, p, heatmap, correct, predicted in zip(images, prompts, heatmaps, target_boxes, prediction_boxes):
            size = img.size
            img_arr = np.asarray(img)
            draw_box_on_image(img_arr, size, predicted, (255,0,0))                    
            draw_box_on_image(img_arr, size, correct, (0,255,0))

            f = plt.figure(figsize=(12,6))
            plt.title(p)       
            plt.axis("off") 
            ax=f.add_subplot(1, 2, 1)
            ax.imshow(img_arr)
            ax.axis("off")
            ax=f.add_subplot(1, 2, 2)
            ax.imshow(heatmap.cpu())
            ax.axis("off")
            if save:
                plt.savefig(f"imgs/{c}.png")
            plt.show()
            c += 1 
            
        print(f"correct {correct}, predict: {predicted}")

    loss = loss_fn(prediction_boxes, target_boxes)
    return loss.item(), iou(prediction_boxes, target_boxes).item()

def training_routine(model, loss_fn, feature_extractor, optimizer):
    model.train()   
    epoch_loss =[]
    giou = []
    for i in tqdm(range(0, train_size, train_batch_size)):
        optimizer.zero_grad()

        batch_data = mini_train_data[i:i+train_batch_size]
        images, prompts, target_boxes, target_heatmaps = get_batch_data(batch_data)  
        with torch.no_grad():
            pred_heatmaps = feature_extractor(images, prompts)     
        prediction_boxes = model(pred_heatmaps.to(device))

        loss = loss_fn(prediction_boxes, target_boxes)      

        epoch_loss.append(loss.item())
        giou.append(iou(prediction_boxes, target_boxes).item())
        loss.backward()
        optimizer.step()

    return sum(epoch_loss) / len(epoch_loss), sum(giou) / len(giou)

def validation_routine(model, loss_fn, feature_extractor):
    model.eval()   
    epoch_loss =[]
    giou = []
    for i in tqdm(range(0, validation_size, validation_batch_size)):
        batch_data = mini_val_data[i:i+validation_batch_size]
        images, prompts, target_boxes, target_heatmaps = get_batch_data(batch_data)  
        with torch.no_grad():
            pred_heatmaps = feature_extractor(images, prompts)     
            prediction_boxes = model(pred_heatmaps.to(device))
            loss = loss_fn(prediction_boxes, target_boxes)      
            epoch_loss.append(loss.item())
            giou.append(iou(prediction_boxes, target_boxes).item())

    return sum(epoch_loss) / len(epoch_loss), sum(giou) / len(giou)

def finetune_clip_routine(loss_fn, feature_extractor, optimizer):
    epoch_loss =[]
    for i in range(0, train_size, train_batch_size):
        optimizer.zero_grad()
        batch_data = mini_train_data[i:i+train_batch_size]
        images, prompts, target_boxes, target_heatmaps = get_batch_data(batch_data)  
        heatmaps = feature_extractor(images, prompts) 
        rem_loss = loss_fn(heatmaps, target_heatmaps)
        epoch_loss.append(rem_loss.item())

        rem_loss.backward()
        optimizer.step()

    return sum(epoch_loss) / len(epoch_loss)


# ### Training Cycle

# #### Finetuning the Spatial CLIP

# In[ ]:


feature_extractor = ResNetHighResV2(clip_preprocess, clip_model, clip.tokenize, remap_heatmaps=True, temperature=0.35).to(device)


# In[ ]:


loss_fn = nn.MSELoss()
finetune = False
if finetune:
    optimizer = torch.optim.Adam(params=feature_extractor.spatial_model.visual.attnpool.parameters(), lr=4E-6)
    feature_extractor.eval()
    feature_extractor.spatial_model.visual.attnpool.train()
    for epoch in range(epochs):
        loss = finetune_clip_routine(loss_fn ,feature_extractor, optimizer)
        print(f"epoch {epoch}")
        print(f"loss: {loss}")             
    optimizer.zero_grad(set_to_none=True)
    feature_extractor.eval()
    torch.save(feature_extractor, "extractor.pt")


# In[ ]:


#feature_extractor = torch.load("extractor.pt")


# #### Training the box regressor
# 
# > We finished available hours of the Azure server quite fast, due to testing and training of old pipelines.
# 
# The following training process has been carried out on our personal PC, therefore we were forced to use very small training set.
# As can be seen the model overfits the data very fast, due to the fact that the training data is very small, around 1000 elements. 
# 
# The overfit point is dependent of mainly the dimension of the dataset.
# We believe that in situations in which the dataset is successfully large, the model could be able to effectively represent data, as intentionally overfitting small dataset allowed us to achieve under `0.5` GIoU.
# 
# In the testing environment, the model reached around `0.8` GIoU by only training on a dataset of 1000 elements.
# 
# In order to improve the current pipeline, suggestions are presented at the end of the notebook, in the Future Work section.
# 
# ![image-2.png](attachment:image-2.png)

# In[ ]:


bboxer = HeatmapToBox().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=bboxer.parameters(), lr=4E-4)


# In[ ]:


best = 1E3
best_giou = 1E4
val_loss = val_giou = 1E3
with open("training.csv", "w", newline="") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(["epoch", "loss", "giou","val_loss","val_giou"])

for epoch in range(epochs):
    # if epoch % 2 == 0:
    #     val_loss, val_giou = validation_routine(bboxer, loss_fn, feature_extractor)
    loss, giou = training_routine(bboxer, loss_fn, feature_extractor, optimizer)
    with open("training.csv", "a", newline="") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow([epoch, loss, giou, val_loss, val_giou])

    print(f"epoch {epoch} | loss {loss} | giou {giou} | val_loss {val_loss} | val_giou {val_giou}")
    
    # if best_giou > val_giou:
    #     torch.save(bboxer, "checkpoint.pt")
    #     best_giou=val_giou     
torch.save(bboxer, "regressor.pt")


# In[ ]:


# %matplotlib inline
# df = pd.read_csv("training.csv")

# f = plt.figure(figsize=(12,6))

# ax=f.add_subplot(1, 2, 1)
# ax.set_title("MSELoss")
# ax.plot(df.epoch, df.loss)
# ax.plot(df.epoch, df.val_loss.rolling(10).mean())
# ax.legend(["training","validation"], loc=3)

# ax=f.add_subplot(1, 2, 2)
# ax.set_title("GIoU")
# ax.plot(df.epoch, df.giou)
# ax.plot(df.epoch, df.val_giou.rolling(10).mean())
# ax.legend(["training","validation"], loc=3)

# plt.savefig("losses.png")
# plt.show()


# **Save the model manually**

# In[ ]:


#torch.save(bboxer, "regressor.pt")


# **Load the model**

# In[ ]:


#bboxer = torch.load("regressor.pt")


# ## Testing and evaluating
# 
# We noticed that the `HeatmapToBox` model is **highly dependant** on the quality of the heatmaps generated. If the heatmap is good, the training process is extremely faster and more effective.
# 
# This means that extensively training the Spatial CLIP model on the dataset would result is extremely better performance.
# 
# We didn't have the possibility to train the Spatial CLIP model on the dataset due to having finished the available hours of the Azure's server. This means that the regressor is extremely subjective to overfitting since we trained on very smallt raining data on our personal computers.
# 
# Nonetheless, the below results have been achieved on the training data without training the Spatial CLIP at all.
# 
# Although the examples are drawn from the data the model is trained on, it can be seen that the model is able to **effectively represent** the data, suggesting that further regressor training togheter with the training of the Spatial CLIP could result in very good performance.
# 
# -----
# 
# > The presented results have been achieved on the training set
# 
# As the results show, the model is able to represent the data, we tested by intetionally overfitting small datasets.
# 
# - MSELoss: `5.7E-3`
# - GIoU : `0.51`
# 
# -----
# 
# ![14.png](attachment:14.png)
# ![12.png](attachment:12.png)
# ![8.png](attachment:8.png)
# ![15.png](attachment:15.png)

# In[ ]:


def evaluate(start, end, batch_size, data, graphical=False):
    l = []
    io = []
    for i in tqdm(range(start, end, batch_size)):
        batch_data = data[i:i+batch_size]
        loss, giou = evaluate_batch_routine(bboxer, loss_fn, feature_extractor, batch_data, graphical=graphical)
        l.append(loss)
        io.append(giou)
    print(f"loss: {sum(l)/len(l)}, iou: {sum(io)/len(io)}")


# Run on the test set:

# In[ ]:


evaluate(0, len(test_data), 16, test_data)


# Run on the whole dataset

# In[ ]:


whole = train_data + validate_data + test_data
evaluate(0, len(whole), 32, whole)


# # Conclusions and Future Work
# 
# The proposed model is interesting due to the following reasons:
# - It is a **single stage model** to perform phrase grounding
# - It enables the CLIP model to be used for phrase grounding
# - The modularity of the framework enabled the head to be swapped extremely easily. This means that any head capable of regressing a bounding box from an heatmap in the proposed form can be attached
# - It involves the possibility to finetune CLIP itself for the phrase grounding task
# 
# The quite poor validation performance achieved is due to many reasons. 
# 
# - The training data has not been exploited at its fullest: due to time and computational power constraints, an extensive process of training could not be achieved
# - The regressor is a brand new model that does not exploit transfer learning, meaning that it has to be trained from scratch
# 
# Nonetheless, the model was able to perfectly overfit small samples of the training data, demonstrating that its strucutre is able to actually represent the data samples in an efficient way.
# 
# #### Problems
# 
# We found that the model required a lot of time to be trained, we finished very fast our time available on the Azure server.
# 
# Also the problem of overfitting was quite high in our model. The reasons may be reconducted to the fact that our head does not include transfer learning and must be trained from scratch. Also the size of the dataset we were able to train the model on played an important role.
# 
# We tried implementing some techniques such as `Dropout` layers and `BarchNorm` layers. The situation remained almost the same.
# 
# #### Improvements
# 
# The model could be improved in the following ways:
# - Extensive training on larger dataset, with better hardware resources
# - Tweaks to the shape and the hyperparameters of the `HeatmapToBox` head model
# - Finetuning the Spatial CLIP model itself in order to better adapt it to the target dataset
# 
# Data augmentation has been tested. However, we believe that it may not be the most crucial point in fixing the overfitting problem as the RefCOCOg dataset includes different prompts for the same image and bounding box. This can be effectively already be seen as a text augmentation. From the point of view of the model, the same image with different prompts, must produce the same bounding box.
# 
# 
# The results showed that potentially this structure may serve as a good fundation for phrase grounding framework that are able to adapt pretrained larger models such as CLIP for a specific task such as phrase grounding.

# # Bibliography
# - [1] [Adapting CLIP For Phrase Localization Without Further Training](https://arxiv.org/abs/2204.03647)
# - [2] [A Fast and Accurate One-Stage Approach to Visual Grounding](https://arxiv.org/abs/1908.06354)
# - [3] [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)
# - [4] [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
