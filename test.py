import os, glob
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from unet_base import UNet, center_crop

class ISICTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_tf, mask_tf):
        self.imgs  = sorted(glob.glob(os.path.join(img_dir,"*.jpg")))
        self.masks = sorted(glob.glob(os.path.join(mask_dir,"*.png")))
        self.img_tf, self.mask_tf = img_tf, mask_tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        img  = Image.open(self.imgs[i]).convert("RGB")
        mask = Image.open(self.masks[i]).convert("L")
        img  = self.img_tf(img)
        mask = self.mask_tf(mask)
        return img, (mask>0).float()

def dice_coeff(pred, tgt, eps=1e-6):
    p=tgt.new_tensor(pred.view(pred.size(0),-1))
    t=tgt.view(tgt.size(0),-1)
    inter=(p*t).sum(1); union=p.sum(1)+t.sum(1)
    return ((2*inter+eps)/(union+eps)).mean().item()

def iou_score(pred, tgt, eps=1e-6):
    p=tgt.new_tensor(pred.view(pred.size(0),-1))
    t=tgt.view(tgt.size(0),-1)
    inter=(p*t).sum(1); union=p.sum(1)+t.sum(1)-inter
    return ((inter+eps)/(union+eps)).mean().item()

def main():
    TEST_IMG, TEST_MASK = "./test/images", "./test/masks"
    INPUT, OUT = 256, 68
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    mask_tf= transforms.Compose([transforms.CenterCrop((OUT,OUT)),
                                 transforms.ToTensor()])

    ds = ISICTestDataset(TEST_IMG, TEST_MASK, img_tf, mask_tf)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = UNet(3,1).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    total_d, total_i = 0,0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = (torch.sigmoid(model(imgs))>0.5).float()
            if preds.shape!=masks.shape:
                masks = center_crop(masks, preds)
            total_d += dice_coeff(preds, masks)
            total_i += iou_score(preds, masks)
    n = len(loader)
    print(f"Test Dice: {total_d/n:.4f}  Test IoU: {total_i/n:.4f}")

if __name__=="__main__":
    main()
