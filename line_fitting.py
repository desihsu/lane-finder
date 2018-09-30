import numpy as np
import cv2


# Finds lane pixels using sliding windows
def find_lane_pixels(warped, nwindows=9, margin=100, minpix=80):
    window_height = np.int(warped.shape[0] // nwindows)
    hist = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    mid = np.int(hist.shape[0]//2)

    # Base and current positions for left and right lines
    leftx_base = np.argmax(hist[:mid])
    rightx_base = np.argmax(hist[mid:]) + mid
    left_lane_inds, right_lane_inds = [], []
    leftx_current, rightx_current = leftx_base, rightx_base

    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    for window in range(nwindows):
        # Window boundaries for left and right lines
        y_low = warped.shape[0] - (window+1)*window_height
        y_high = warped.shape[0] - window*window_height
        leftx_low = leftx_current - margin
        leftx_high = leftx_current + margin
        rightx_low = rightx_current - margin
        rightx_high = rightx_current + margin

        # Nonzero pixels in x and y within window
        good_left_inds = ((nonzerox >= leftx_low) & 
                          (nonzerox < leftx_high) & 
                          (nonzeroy >= y_low) & 
                          (nonzeroy < y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= rightx_low) & 
                           (nonzerox < rightx_high) & 
                           (nonzeroy >= y_low) & 
                           (nonzeroy < y_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Extracts left and right line pixel positions
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty


# Fits a polynomial to lane line pixels
def fit_polynomial(warped, detected=True, leftx=None, lefty=None, 
                   rightx=None, righty=None):
    # Uses sliding windows if no previous polynomial fit available
    if not detected:
        leftx, lefty, rightx, righty = find_lane_pixels(warped)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generates x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fit, right_fit, left_fitx, right_fitx, ploty


def search_around_poly(warped, left_fit, right_fit, margin=100):
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Area of search within margins of previous polynomial fit
    poly_left = (left_fit[0]*(nonzeroy**2) + 
                 left_fit[1]*nonzeroy + left_fit[2])
    poly_right = (right_fit[0]*(nonzeroy**2) + 
                  right_fit[1]*nonzeroy + right_fit[2])
    left_lane_inds = ((nonzerox > poly_left - margin) & 
                     (nonzerox < poly_left + margin))
    right_lane_inds = ((nonzerox > poly_right - margin) & 
                      (nonzerox < poly_right + margin))

    # Extracts left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial(
            warped, detected=True, leftx=leftx, lefty=lefty, 
            rightx=rightx, righty=righty)
    return left_fit, right_fit, left_fitx, right_fitx, ploty


def draw_lines(img, warped, Minv, left_fitx, right_fitx, ploty):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recasts x and y points
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(
                                np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw lanes and warp back to original
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
    new_warp = cv2.warpPerspective(color_warp, Minv, 
                                   (img.shape[1],img.shape[0])) 
    result = cv2.addWeighted(img, 1, new_warp, 0.3, 0)
    return result