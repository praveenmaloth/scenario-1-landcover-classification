import pandas as pd
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, df, root, class2idx, transform):
        self.df = df.reset_index(drop=True); self.root=Path(root); self.transform=transform; self.class2idx=class2idx
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row=self.df.iloc[idx]; img=Image.open(self.root/row['filename']).convert('RGB')
        return self.transform(img), self.class2idx[row['class_name']], row['filename']

# usage: python scripts/evaluate.py --model outputs/best_resnet18.pth --test_csv outputs/test_split.csv --rgb_dir data/rgb
