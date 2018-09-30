import numpy as np
import matplotlib.image as mpimg
import cv2
import os


# Camera calibration using chessboard corners
def calibrate(nx=9, ny=6, img_path="camera_calibration/"):
    # Replicates array of coordinates
    objpoints, imgpoints = [], []
    objpts = np.zeros((nx*ny, 3), np.float32)
    objpts[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for name in os.listdir(img_path):
        img = mpimg.imread(img_path + name)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            objpoints.append(objpts)
            imgpoints.append(corners)

    # Distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                       img.shape[1::-1], 
                                                       None, None)
    return mtx, dist


def perspective_transform(img, mtx, dist):
    h, w = img.shape[0], img.shape[1]
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Perspective transform matrices
    src = np.float32([[220,h], [1100,h],[720,470], [570,470]])
    dst = np.float32([[320,h], [920,h], [920,0], [320,0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped, Minv