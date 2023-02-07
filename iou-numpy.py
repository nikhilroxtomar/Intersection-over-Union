
import numpy as np
import cv2

def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    # return the intersection over union value
    return iou

boxA = [100, 100, 250, 250]
boxB = [150, 150, 300, 300]

iou = intersection_over_union(boxA, boxB)
print("IoU: ", iou)

# A simple 400 x 400 blnk white image
image = np.ones((400, 400, 3), dtype=np.float32) * 255

# Plotting IoU on image
cv2.putText(image, f"IoU: {iou:.4f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Plotting bounding box A
image = cv2.rectangle(image, (boxA[0], boxA[1]), (boxA[2], boxA[3]), (255, 0, 0), 2) ## Blue
cv2.putText(image, "Box A", (boxA[0], boxA[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Plotting bounding box B
image = cv2.rectangle(image, (boxB[0], boxB[1]), (boxB[2], boxB[3]), (0, 0, 255), 2) ## Red
cv2.putText(image, "Box B", (boxB[0], boxB[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imwrite("iou.png", image)
