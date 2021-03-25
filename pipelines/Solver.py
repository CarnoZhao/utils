seed = 0
lr = 1e-4
n_epochs = 20
n_classes = 15
batch_size = 32
size = 224
drop = 0.5
alpha = 0
n_folds = 4
name = "vit_large_patch16_224"
singlefold = 0
foldstop = 0
gpus = "0,1"
fold = 0; epoch = 1;

import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
import warnings
warnings.filterwarnings("ignore")

import re
import cv2
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from apex import amp
import albumentations as A
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchsampler import ImbalancedDatasetSampler

from utils.mixup import mixup_data, mixup_criterion
from utils.loss.smooth import LabelSmoothingLoss
from utils import make_file_path, setup_seed, Log
modelpath, plotpath, outpath, starttime, basepath = make_file_path(__file__)
setup_seed(seed)
logger = print if "outpath" not in locals() else Log(locals()["outpath"])
logger(starttime)

image_dir = f"./data/train256"
dfs = []
for i, label in enumerate(sorted(os.listdir(image_dir))):
    # os.mkdir(os.path.join("./data/train256", label))
    # for f in os.listdir(os.path.join(image_dir, label)):
    #     img = Image.open(os.path.join(image_dir, label, f)).convert("RGB").resize((256, 256))
    #     img.save(os.path.join("./data/train256", label, f.replace("jpg", "png")))
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

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.criterion = LabelSmoothingLoss(n_classes, 0.1)

    def forward(self, logits, target):
        loss = self.criterion(logits, target)
        return loss

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model(name, pretrained = True, in_chans = 3, num_classes = n_classes)
        self.model.head = nn.Sequential(
            nn.Dropout(drop),
            self.model.head
        )
        # classifier for effnet
        # head for vit
        # fc for resnet
        
    def forward(self, x):
        return self.model(x)

def model_init():
    model = Model()
    model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-2)
    criterion = Criterion()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, pct_start = 0.1, div_factor = 25, epochs = n_epochs, steps_per_epoch = len(dl_train))
    
    model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids = list(range(len(gpus.split(",")))))
    return model, optimizer, criterion, scheduler

def train_epoch(dl_train, args):
    model, optimizer, criterion, scheduler = args
    model.train()
    train_loss = []
    for batch_idx, (x, y) in enumerate(dl_train):
        x = x.cuda()
        y = y.cuda()
        if alpha != 0:
            xm, ya, yb, lam = mixup_data(x, y, alpha)
            yhat = model(xm)
            loss = mixup_criterion(criterion, yhat, ya, yb, lam)
        else:
            yhat = model(x)
        loss = criterion(yhat, y)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.detach().cpu().numpy())
        scheduler.step()
    return np.nanmean(train_loss)

def val_epoch(dl_valid, args, is_last = False):
    model, _, criterion, _ = args
    model.eval()
    val_loss = []
    TARGET = []
    LOGITS = []
    with torch.no_grad():
        for x, y in dl_valid:
            x = x.cuda()
            xs = [x]# , x.flip(-1), x.flip(-2), x.flip(-1, -2), 
                #   x.transpose(-1, -2), x.transpose(-1, -2).flip(-1), 
                #   x.transpose(-1, -2).flip(-2), x.transpose(-1, -2).flip(-1, -2)]
            y = y.cuda()
            yhat = None
            for x in xs:
                yhat = model(x) if yhat is None else (yhat + model(x))
            yhat /= len(xs)
            loss = criterion(yhat, y)
            val_loss.append(loss.detach().cpu().numpy())
            TARGET.append(y.detach().cpu().numpy())
            LOGITS.append(yhat.argmax(1).detach().cpu().numpy())
    TARGET = np.concatenate(TARGET)
    LOGITS = np.concatenate(LOGITS)
    mets = [
        f1_score(TARGET, LOGITS, average = "macro"),
        accuracy_score(TARGET, LOGITS),
        precision_score(TARGET, LOGITS, average = "macro"),
        recall_score(TARGET, LOGITS, average = "macro")
    ]
    if is_last:
        logger(confusion_matrix(TARGET, LOGITS))
    return np.nanmean(val_loss), mets

train_trans = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.ShiftScaleRotate(scale_limit = 0.2, rotate_limit = 20, p = 0.9, border_mode=cv2.BORDER_REFLECT),
    A.OneOf([
        A.IAAPiecewiseAffine(),
        A.GridDistortion(),
        A.OpticalDistortion(),
    ], p = 1),
    A.OneOf([
        A.RandomBrightnessContrast(),
        A.HueSaturationValue(),
        A.CLAHE()
    ], p = 1),
    A.OneOf([
        A.RandomRain(),
        A.RandomSnow(),
        A.RandomFog()
    ], p = 1),
    A.Resize(size, size),
    A.Normalize()
], p = 1)
valid_trans = A.Compose([
    A.Resize(size, size),
    A.Normalize()
])

split = StratifiedKFold(n_folds, shuffle = True, random_state = seed)
model = None
for fold in range(fold, n_folds):
    METRIC = 0.
    logger(f"\n*******START: seed {seed}, fold {fold}*******")

    train_idx, valid_idx = list(split.split(df, df.label))[fold]
    df_train = df.iloc[train_idx] if not singlefold else df.copy()
    df_valid = df.iloc[valid_idx]
    ds_train = Data(df_train, train_trans)
    ds_valid = Data(df_valid, valid_trans)
    # sampler = ImbalancedDatasetSampler(ds_train, callback_get_label = lambda x, i: x.df.label.iloc[i])
    # dl_train = DataLoader(ds_train, batch_size = batch_size, sampler = sampler, num_workers = 4)
    dl_train = DataLoader(ds_train, batch_size = batch_size, shuffle = True, num_workers = 4)
    dl_valid = DataLoader(ds_valid, batch_size = batch_size, shuffle = False, num_workers = 4)
    logger(f"bs: {batch_size}, ds_train: {len(ds_train)}, df_valid: {len(df_valid)}")

    args = model_init()
    model, optimizer, criterion, scheduler = args
    
    for epoch in range(1, n_epochs + 1):
        t1 = time.time()
        train_loss = train_epoch(dl_train, args)
        t2 = time.time()
        val_loss, mets = val_epoch(dl_valid, args, epoch == n_epochs)
        t3 = time.time()

        content = time.strftime("%H:%M:%S", time.localtime()) + f'|EP {epoch:2d}|T1 {round(t2 - t1)}|T2 {round(t3 - t2)}|LR {optimizer.param_groups[0]["lr"]:.1e}|L1 {train_loss:.3f}|L2 {val_loss:.3f}|AC {mets[1]:.3f}|PR {mets[2]:.3f}|RE {mets[3]:.3f}|F1 {mets[0]:.3f}'
        logger(content)

        metric = mets[0]
        if metric > METRIC:
            logger('BEST METRIC {:.6f} --> {:.6f}'.format(METRIC, metric))
            METRIC = metric
            torch.save((model.module if len(gpus) > 1 else model).state_dict(), modelpath.replace(".pt", f"/best/{starttime.replace(':', '_').replace('.', '_')}_{fold}.pt"))
    logger("\n********END********\n")

    torch.save((model.module if len(gpus) > 1 else model).state_dict(), modelpath.replace(".pt", f"/final/{starttime.replace(':', '_').replace('.', '_')}_{fold}.pt"))
    if foldstop or singlefold: break
