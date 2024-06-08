import torch
import torch.nn as nn

from utils.iou import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(self, YoloLoss).__init__()

        self.mse = nn.MSELoss(reduction= "sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, prediction, target):
        prediction = prediction.reshape(-1, self.S, self.S, self.C + self.B*5)

        iou_b1  = intersection_over_union(prediction[...,21:25], target[...,21:25])
        iou_b2  = intersection_over_union(prediction[...,26:30], target[...,21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes ,best_box = torch.max(ious, dim=0)
        exists_box = target[...,20].unsqueeze(3) #Iobj I it will tell i there is an object in cell i

        ####  For Box Coordinates  ####
        box_predictions = exists_box * ((
            best_box * prediction[..., 26:30] + (1 - best_box) * prediction[...,21:25]))

        box_targets = exists_box * target[...,21:25]

        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(torch.abs(box_predictions[...,2:4] + 1e-6))

        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

        # (N, S, S, 4) -> (N * S * S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2),
            torch.flatten(box_targets, end_dim = -2)
        )

        ####     For Object Loss   ####
        pred_box = (
            best_box * prediction[...,25:26] + (1 - best_box) * prediction[... ,20:21]
        )

        #(N * S * S * 1)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[...,20:21])

        )

        ####   For No Object Loss  ####
        # (N, S, S, 1) -> (N, S *S )
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * prediction[...,20:21], start_dim = 1),
            torch.flatten(1 - exists_box)  * target[..., 20:21], start_dim = 1
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * prediction[...,25:26], start_dim = 1),
            torch.flatten(1 - exists_box)  * target[..., 20:21], start_dim = 1
        )

        ####    For Class loss     ####
        #(N, S, S, 20) -> (N * S *S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * prediction[...,:20], end_dim = -2),
            torch.flatten(exists_box* target[...,:20], end_dim= -2)
        )
        
        loss = (self.lambda_coord * box_loss  # First two rows of loss in paper
        + object_loss 
        + self.lambda_noobj * no_object_loss
        + class_loss)

        return loss