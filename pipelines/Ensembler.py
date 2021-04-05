seed = 0
n_classes = 15
batch_size = 64
size = 224
name = "vit_large_patch16_224"
gpus = "0"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


image_dir = f"./data/train256"
dfs = []
for i, label in enumerate(sorted(os.listdir(image_dir))):
    df = pd.DataFrame({"image": [os.path.join(image_dir, label, f) for f in os.listdir(os.path.join(image_dir, label))]})
    df["label_name"] = label
    df["label"] = i
    dfs.append(df)
df = pd.concat(dfs).reset_index(drop = True)


class Data(Dataset):
    def __init__(self, df, trans = None):
        self.df = df
        self.trans = trans
  
    def __getitem__(self, index):
        label = np.array(self.df.label.iloc[index])
        image = np.array(Image.open(self.df.image.iloc[index]).convert(mode = "RGB"))
        if self.trans is not None:
            aug = self.trans(image = image)
            image = aug["image"]
        image = image.astype(np.float32).transpose(2, 0, 1)
        return image, label
    
    def __len__(self):
        return self.df.shape[0]

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model(name, pretrained = True, in_chans = 3, num_classes = n_classes)
        self.model.head = nn.Sequential(
            nn.Dropout(0),
            self.model.head
        )
        
    def forward(self, x):
        return self.model(x)

def get_models(models_path):
    models = []
    for model_path in models_path:
        model = Model()
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()
        model.requires_grad_(False)
        models.append(model)
    return models

def val_epoch(dl_valid, models):
    TARGET = []
    LOGITS = []
    for x, y in tqdm(dl_valid):
        x = x.cuda()
        xs = [x , x.flip(-1), x.flip(-2), x.flip(-1, -2), 
              x.transpose(-1, -2), x.transpose(-1, -2).flip(-1), 
              x.transpose(-1, -2).flip(-2), x.transpose(-1, -2).flip(-1, -2)]
        y = y.cuda()
        yhat = None
        for model in models:
            for x in xs:
                yhat = model(x) if yhat is None else (yhat + model(x))
        yhat /= len(xs) * len(models)
        TARGET.append(y.detach().cpu().numpy())
        LOGITS.append(yhat.detach().cpu().numpy())
    TARGET = np.concatenate(TARGET)
    LOGITS = np.concatenate(LOGITS)
    return TARGET, LOGITS

models_path = [
    *[f"_models/Mar131740/best/Mar131740_{i}.pt" for i in range(4)]
]
models = get_models(models_path)

trans = A.Compose([
    A.Resize(size, size),
    A.Normalize()
])
ds = Data(df, trans)
dl = DataLoader(ds, batch_size = batch_size, num_workers = 4)

TARGET, LOGITS = val_epoch(dl, models)

top2_acc = ((LOGITS.argsort(1)[:,-2:] - TARGET[:,np.newaxis]) == 0).any(1).mean(); top2_acc
cm = confusion_matrix(TARGET, LOGITS.argmax(1))