import sys
from moviepy.editor import VideoFileClip
import camera
import image_thresholding
import line_fitting
import matplotlib.image as mpimg


class Line():
    def __init__(self):
        self.detected = False  # lane line detected in previous iteration
        self.fit = None  # most recent polynomial fit
        self.fitx = None  # most recent x pixel values for line


def process_image(img):
    color_grad_combined = image_thresholding.color_grad(img)
    warped, Minv = camera.perspective_transform(color_grad_combined, mtx, dist)

    if left_line.detected and right_line.detected:
        (left_line.fit, right_line.fit, 
         left_line.fitx, right_line.fitx, 
         ploty) = line_fitting.search_around_poly(warped, left_line.fit, 
                                                  right_line.fit)
    else:
        (left_line.fit, right_line.fit, 
         left_line.fitx, right_line.fitx, 
         ploty) = line_fitting.fit_polynomial(warped, detected=False)
        left_line.detected = True
        right_line.detected = True

    result = line_fitting.draw_lines(img, warped, Minv, left_line.fitx, 
                                     right_line.fitx, ploty)
    return result


if __name__ == "__main__":
    mtx, dist = camera.calibrate()
    left_line = Line()
    right_line = Line()

    if (sys.argv[1].split(".")[-1] == "mp4"):
    	clip = VideoFileClip(sys.argv[1])
    	output = clip.fl_image(process_image)
    	output.write_videofile("output.mp4", audio=False)
    else:
    	img = mpimg.imread(sys.argv[1])
    	img = process_image(img)
    	mpimg.imsave("output.jpg", img)