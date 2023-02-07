import tensorflow as tf

def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = tf.maximum(boxA[0], boxB[0])
    y1 = tf.maximum(boxA[1], boxB[1])
    x2 = tf.minimum(boxA[2], boxB[2])
    y2 = tf.minimum(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    intersection_area = tf.maximum(0, x2 - x1 + 1) * tf.maximum(0, y2 - y1 + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / (boxA_area + boxB_area - intersection_area)

    # return the intersection over union value
    return iou

boxA = [100, 100, 250, 250]
boxB = [150, 150, 300, 300]

iou = intersection_over_union(boxA, boxB)
print("IoU: ", iou)
