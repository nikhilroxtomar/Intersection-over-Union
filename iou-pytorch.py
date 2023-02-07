import torch

def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = torch.max(boxA[0], boxB[0])
    y1 = torch.max(boxA[1], boxB[1])
    x2 = torch.min(boxA[2], boxB[2])
    y2 = torch.min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    intersection_area = torch.max(torch.tensor(0.0), x2 - x1 + 1) * torch.max(torch.tensor(0.0), y2 - y1 + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / (boxA_area + boxB_area - intersection_area)

    # return the intersection over union value
    return iou

boxA = torch.tensor([100, 100, 250, 250])
boxB = torch.tensor([150, 150, 300, 300])

iou = intersection_over_union(boxA, boxB)
print("IoU: ", iou)
