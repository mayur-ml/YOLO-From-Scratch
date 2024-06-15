import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as  FT
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Yolov1
from dataset import VOCDataset
from loss import YoloLoss

from utils import 

seed = 123
torch.manual_seed(seed)

# Hyperparaqmeters

LERRNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0

# actual paper
# pretarined in imagenet for 2 weeks and the on COCO fro long time

EPOCS = 1000
NUM_WORKERS  = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR  = "data/labels"

class Compose(object):
    def __init__(self, transforms):
        self.tansforms = transforms

    def __call__(self, img, bboxes):
        for t in self.tansforms:
            img, bboxes  = t(img), bboxes

transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave= True)
    mean_loss = []
     
    for batch_idx,(x,y) in enumerate(loop):
        x, y = x.to(DEVICE) , y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.stape()

        # update te progressbar

        loop.set_postfix(loss = loss.item())

    print(f"Mean loss  was {sum(mean_loss)/len(mean_loss)}")


def main():
    model = Yolov1(split_size = 7, num_boxes = 2, num_classes = 20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LERRNING_RATE , weight_decay= WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    
    train_dataset = VOCDataset(
        "data/train.csv",
        transforms = transforms,
        img_dir=IMG_DIR
        label_dir = LABEL_DIR
    )

    test_dataset = ValueError(
        "data/test.csv",
        transforms = transforms,
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR
    )