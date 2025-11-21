import argparse
import pandas as pd
from pathlib import Path
import torch
from torchvision import transforms, models
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# minimal dataset class
class SentinelDataset(Dataset):
    def __init__(self, df, root, class2idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.class2idx = class2idx
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row['filename']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.class2idx[row['class_name']]
        return img, label

def main(args):
    outputs = Path(args.outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(args.train_csv)
    classes = sorted(train_df['class_name'].unique())
    class2idx = {c:i for i,c in enumerate(classes)}

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    ds = SentinelDataset(train_df, args.rgb_dir, class2idx, transform=transform)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        model.train()
        running=0
        for imgs, labels in loader:
            imgs = imgs.to(device); labels = labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward(); opt.step()
            running += loss.item()*imgs.size(0)
        print(f"Epoch {epoch} loss {running/len(ds):.4f}")
        torch.save(model.state_dict(), outputs/'last.pth')
    torch.save(model.state_dict(), outputs/'best_resnet18.pth')

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True)
    p.add_argument('--rgb_dir', required=True)
    p.add_argument('--outputs', default='outputs')
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()
    main(args)