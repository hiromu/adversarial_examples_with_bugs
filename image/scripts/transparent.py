#!/usr/bin/env python

import cv2
import numpy as np
import os
import sys

DIFF = (3, 3, 3)
MEDIAN = 5


def transparent(imgpath):
    img = cv2.imread(imgpath)
    h, w = img.shape[:2]

    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 8 | 255 << 8 | cv2.FLOODFILL_MASK_ONLY

    for point in ((5, 5), (5, w - 5), (h - 5, 5), (h - 5, w - 5)):
        submask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(img, submask, seedPoint=(5, 5), newVal=(0, 0, 255), loDiff=DIFF, upDiff=DIFF, flags=flags)
        mask |= submask

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask = mask[1:-1, 1:-1]
    rgba[mask==255] = 0

    median = cv2.medianBlur(rgba, MEDIAN)
    filename, _ = os.path.splitext(os.path.basename(imgpath))
    cv2.imwrite(filename + '.png', median)


if __name__ == '__main__':
    for imgpath in sys.argv[1:]:
        transparent(imgpath)

