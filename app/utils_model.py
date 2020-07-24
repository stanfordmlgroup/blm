from utils_nms import apply_nms
from PIL import Image
import cv2
import torch
import numpy as np
import copy
import logging

def compute_boxes_and_sizes(PRED_DOWNSCALE_FACTORS, GAMMA, NUM_BOXES_PER_SCALE):

    BOX_SIZE_BINS = [1]
    g_idx = 0
    while len(BOX_SIZE_BINS) < NUM_BOXES_PER_SCALE * len(PRED_DOWNSCALE_FACTORS):
        gamma_idx = len(BOX_SIZE_BINS) // (len(GAMMA) - 1)
        box_size = BOX_SIZE_BINS[g_idx] + GAMMA[gamma_idx]
        BOX_SIZE_BINS.append(box_size)
        g_idx += 1

    BOX_SIZE_BINS_NPY = np.array(BOX_SIZE_BINS)
    BOXES = np.reshape(BOX_SIZE_BINS_NPY, (4, 3))
    BOXES = BOXES[::-1]

    return BOXES, BOX_SIZE_BINS


def upsample_single(input_, factor=2):
    channels = input_.size(1)
    indices = torch.nonzero(input_)
    indices_up = indices.clone()
    # Corner case!
    if indices_up.size(0) == 0:
        # return torch.zeros(input_.size(0),input_.size(1), input_.size(2)*factor, input_.size(3)*factor).cuda()
        return torch.zeros(input_.size(0),input_.size(1), input_.size(2)*factor, input_.size(3)*factor)
    indices_up[:, 2] *= factor
    indices_up[:, 3] *= factor

    # output = torch.zeros(input_.size(0),input_.size(1), input_.size(2)*factor, input_.size(3)*factor).cuda()
    output = torch.zeros(input_.size(0),input_.size(1), input_.size(2)*factor, input_.size(3)*factor)
    output[indices_up[:, 0], indices_up[:, 1], indices_up[:, 2], indices_up[:, 3]] = input_[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

    output[indices_up[:, 0], channels-1, indices_up[:, 2]+1, indices_up[:, 3]] = 1.0
    output[indices_up[:, 0], channels-1, indices_up[:, 2], indices_up[:, 3]+1] = 1.0
    output[indices_up[:, 0], channels-1, indices_up[:, 2]+1, indices_up[:, 3]+1] = 1.0

    # output_check = nn.functional.max_pool2d(output, kernel_size=2)

    return output


def get_upsample_output(model_output, output_downscale):
    upsample_max = int(np.log2(16 // output_downscale))
    upsample_pred = []
    for idx, out in enumerate(model_output):
        out = torch.nn.functional.softmax(out, dim=1)
        upsample_out = out
        for n in range(upsample_max - idx):
            upsample_out = upsample_single(upsample_out, factor=2)
        upsample_pred.append(upsample_out.cpu().data.numpy().squeeze(0))
    return upsample_pred


def box_NMS(predictions, nms_thresh, BOXES):
    Scores = []
    Boxes = []
    for k in range(len(BOXES)):
        scores = np.max(predictions[k], axis=0)
        boxes = np.argmax(predictions[k], axis=0)
        # index the boxes with BOXES to get h_map and w_map (both are the same for us)
        mask = (boxes < 3)  # removing Z
        boxes = (boxes + 1) * mask
        scores = (scores * mask)  # + 100 # added 100 since we take logsoftmax and it's negative!!

        boxes = (boxes == 1) * BOXES[k][0] + (boxes == 2) * BOXES[k][1] + (boxes == 3) * BOXES[k][2]
        Scores.append(scores)
        Boxes.append(boxes)

    x, y, h, w, scores = apply_nms(Scores, Boxes, Boxes, 0.5, thresh=nms_thresh)

    nms_out = np.zeros((predictions[0].shape[1], predictions[0].shape[2]))  # since predictions[0] is of size 4 x H x W
    box_out = np.zeros((predictions[0].shape[1], predictions[0].shape[2]))  # since predictions[0] is of size 4 x H x W
    for (xx, yy, hh) in zip(x, y, h):
        nms_out[yy, xx] = 1
        box_out[yy, xx] = hh

    assert (np.count_nonzero(nms_out) == len(x))

    return nms_out, box_out


def get_box_and_dot_maps(pred, nms_thresh, BOXES):
    assert (len(pred) == 4)
    # NMS on the multi-scale outputs
    nms_out, h = box_NMS(pred, nms_thresh, BOXES)
    return nms_out, h


def get_boxed_img(image, original_emoji, h_map, w_map, gt_pred_map, prediction_downscale, BOXES, BOX_SIZE_BINS,
                  thickness=1, multi_colours=False):

    if image.shape[2] != 3:
        boxed_img = image.astype(np.uint8).transpose((1, 2, 0)).copy()
    else:
        boxed_img = image.astype(np.uint8).copy()
    head_idx = np.where(gt_pred_map > 0)

    H, W = boxed_img.shape[:2]

    Y, X = head_idx[-2], head_idx[-1]

    # scale to image 
    enlarge_factor = max(((H * W) / (48 ** 2)) // 300, 1)

    for i, (y, x) in enumerate(zip(Y, X)):
        
        h, w = h_map[y, x]*prediction_downscale, w_map[y, x]*prediction_downscale
        scale = ((BOX_SIZE_BINS.index(h // prediction_downscale)) // 3) + 1

        if enlarge_factor > 1:
            h *= enlarge_factor / 2
            w *= enlarge_factor / 2

        expand_w = (0.2 * scale * w) // 2
        expand_h = (0.2 * scale * h) // 2

        y2 = min(int((prediction_downscale * x + w / 2) + expand_w), W)
        y1 = max(int((prediction_downscale * x - w / 2) - expand_w), 0)
        x2 = min(int((prediction_downscale * y + h / 2) + expand_h), H)
        x1 = max(int((prediction_downscale * y - h / 2) - expand_h), 0)
        
        emoji = copy.deepcopy(original_emoji)
        # emoji = original_emoji.copy()
        width = x2 - x1
        height = y2 - y1
        emoji  = cv2.resize(emoji, (height, width))  
        # emoji = emoji.resize((width, height))
        # emoji = np.array(emoji)

        # https://gist.github.com/clungzta/b4bbb3e2aa0490b0cfcbc042184b0b4e

        # Extract the alpha mask of the RGBA image, convert to RGB 
        r,g,b,a = cv2.split(emoji)
        overlay_color = cv2.merge((b,g,r))
        
        # Apply some simple filtering to remove edge noise
        mask = cv2.medianBlur(a,5)
        mask[mask != 255] = 0
        roi = boxed_img[x1:x2, y1:y2]

        # Black-out the area behind the emoji in our original ROI
        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask = cv2.bitwise_not(mask))
        
        # Mask out the emoji from the emoji image.
        img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask = mask)

        # Update the original image with our new ROI
        boxed_img[x1:x2, y1:y2] = cv2.add(img1_bg, img2_fg)


    return boxed_img  