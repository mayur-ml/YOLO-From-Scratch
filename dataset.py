import torch
import os
import pandas as pd
from PIL import Image
import torch.utils

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S =7, B = 2, C = 20, transform = None):
        super(self, )

        self.annotations = pd.read_csv(csv_file)
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        labbel_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(labbel_path) as f:
            for label in f.readline():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()]
                
                boxes.append([class_label, x, y, width, height])


        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_metrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S, *y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )
            if labbel_path[i, j, 20] == 0:
                label_metrix[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cel, y_cell, width_cell, height_cell]
                )
                label_metrix[i, j, 21:25] = box_coordinates
                label_metrix[i, j, class_label] = 1
                
        return image, label_metrix