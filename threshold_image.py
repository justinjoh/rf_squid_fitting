import cv2
import sys

def display_image(filepath):
    """Display raw image located at relative filepath"""
    raw_image = cv2.imread(filepath)

    # Display raw_image
    # ezshow('raw_image', raw_image)

    user_defined_rectangle(raw_image)


def ezshow(titlestring, mat):
    """Just for making displaying images easier"""
    cv2.imshow(titlestring, mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def user_defined_rectangle(raw_image):
    """Returns user ROI 4-tuple of x and y coords of corners
    Format is (x1, x2, y1, y2)"""
    rect2d_ROI = cv2.selectROI(raw_image, fromCenter=False)
    print("ROI selected")
    x1 = input("Enter x start, milliamps: ")
    x2 = input("Enter x end, milliamps: ")
    y1 = input("Enter y start, GHz: ")
    y2 = input("Enter y end, GHz: ")
    corners_tuple = (x1, x2, y1, y2)

    topleft_x = rect2d_ROI[0]
    topleft_y = rect2d_ROI[1]
    width = rect2d_ROI[2]
    height = rect2d_ROI[3]

    user_rectangle_ROI = (topleft_x, topleft_x + width,
                          topleft_y-height, topleft_y)

    return user_rectangle_ROI, corners_tuple

# TODO: have defined region. Now need to threshold and stuff.

def threshold_raw(filepath):
    """Operates on image filename (trivial to change to operate on raw array)"""
    raw_image = cv2.imread(filepath)
    raw_image_gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    thresh = 120 
    # threshold the image
    thresholded_image= cv2.threshold(raw_image_gray, thresh, 255,
    cv2.THRESH_BINARY)[1]
    ezshow("thresholded", thresholded_image)




if __name__ == "__main__" and len(sys.argv) == 2:
    argv_filepath = sys.argv[1]
    actual_filepath = "%s" % argv_filepath
    display_image(argv_filepath)
    threshold_raw(argv_filepath)
