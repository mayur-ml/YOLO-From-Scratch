# Intersection over Union (IoU) Function

This document explains the `intersection_over_union` function, which calculates the IoU between predicted and ground truth bounding boxes.

## Explanation

The function takes three arguments:

- `boxes_preds`: A tensor containing the predicted bounding boxes with shape (BATCH_SIZE, 4).
- `boxes_labels`: A tensor containing the true bounding boxes with shape (BATCH_SIZE, 4).
- `box_format`: A string indicating the format of the bounding boxes. It can be either "midpoint" or "corners".

## Bounding Box Formats

The function supports two bounding box formats:

- **Midpoint:** (`x_center`, `y_center`, `width`, `height`)
- **Corners:** (`x1`, `y1`, `x2`, `y2`)

## Function Steps

1. **Convert to Corner Coordinates:**
    - If `box_format` is "midpoint": Convert boxes from (`x_center`, `y_center`, `width`, `height`) to (`x1`, `y1`, `x2`, `y2`).
    - If `box_format` is "corners": Use the coordinates directly.
2. **Calculate Intersection:** Compute the area of overlap between the predicted and ground truth boxes using the max of starting points and the min of ending points.
3. **Calculate Union:**
    - Calculate the area of each bounding box.
    - Use the formula: `union = box1_area + box2_area - intersection_area`.
4. **Compute IoU:** IoU is the ratio of the intersection area to the union area.

## Example

Here's an example demonstrating the function with `box_format="midpoint"`:

```python
import torch

# Example bounding boxes (BATCH_SIZE = 2)
# Format: (x_center, y_center, width, height)
preds = torch.tensor([[2.0, 3.0, 2.0, 2.0], [4.0, 5.0, 3.0, 3.0]])
labels = torch.tensor([[2.0, 3.0, 2.0, 2.0], [5.0, 6.0, 2.0, 2.0]])

# Calculate IoU
iou = intersection_over_union(preds, labels, box_format="midpoint")
print(iou)
