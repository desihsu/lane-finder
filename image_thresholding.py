import cv2
import numpy as np


def color_grad(img, r_thresh=(200,255), l_thresh=(120,255), 
               s_thresh=(100,255), grad_thresh=(20,100)):
    # Red mask
    r = img[:,:,0]
    r_mask = (r >= r_thresh[0]) & (r <= r_thresh[1])

    # Saturation and lightness masks
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l = hls[:,:,1]
    s = hls[:,:,2]
    l_mask = (l >= l_thresh[0]) & (l <= l_thresh[1])
    s_mask = (s >= s_thresh[0]) & (s <= s_thresh[1])

    # Absolute gradient mask
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    scaled = np.uint8(255*np.absolute(sobel) / np.max(np.absolute(sobel)))
    grad_mask = (scaled >= grad_thresh[0]) & (scaled <= grad_thresh[1])

    # Combined mask
    combined = np.zeros_like(scaled)
    combined[(r_mask & l_mask) & (s_mask | grad_mask)] = 1

    # Region mask
    h, w = img.shape[0], img.shape[1]
    vertices = np.array([[(0.15*w,h),(0.45*w,0.6*h),
                          (0.55*w,0.6*h),(0.9*w,h)]], dtype=np.int32) 
    region_mask = np.zeros_like(combined)
    cv2.fillPoly(region_mask, vertices, 1)
    result = cv2.bitwise_and(combined, region_mask)
    return result