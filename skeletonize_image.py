import cv2
import sys
import numpy as np

def display_image(filepath):
    """Display raw image located at relative filepath"""
    raw_image = cv2.imread(filepath)

    # Display raw_image
    # ezshow('raw_image', raw_image)

    user_defined_rectangle(raw_image)


def ezshow(titlestring, mat):
    """Just for making displaying images easier"""
    print("Displaying " + str(titlestring)) 
    cv2.imshow(titlestring, mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def user_defined_rectangle(filepath, raw_image=None):
    """Returns user ROI 4-tuple of x and y coords of corners
    Format is (x1, x2, y1, y2)"""
    if not (raw_image == None):
        raw_image = raw_image
    else:
        raw_image = cv2.imread(filepath)
    rect2d_ROI = cv2.selectROI(raw_image, fromCenter=False)
    print("ROI selected")
    x1 = float(raw_input("Enter x start, in phi0: "))
    x2 = float(raw_input("Enter x end, in phi0: "))
    y1 = float(raw_input("Enter y start, GHz: "))
    y2 = float(raw_input("Enter y end, GHz: "))
    corners_tuple = (x1, x2, float(y1)*1e9, float(y2)*1e9)

    return rect2d_ROI, corners_tuple

# TODO: have defined region. Now need to threshold and stuff.

def threshold_raw(filepath, thresh_val, raw_image_gray_arg=None):
    """Operates on image filename (trivial to change to operate on raw array)"""
    if not (raw_image_gray_arg == None):
        raw_image_gray = raw_image_gray_arg
    else:
        raw_image = cv2.imread(filepath)
        raw_image_gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    # threshold the image
    thresholded_image= cv2.threshold(raw_image_gray, thresh_val, 255, cv2.THRESH_BINARY)[1]
    # thresholded_image = cv2.adaptiveThreshold(raw_image_gray, thresh_val,
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 1)
        # TODO: change blocksize (is 15 right now)
    return thresholded_image

def user_defined_threshold(filepath, raw_image_gray_arg=None):
    """Allows user to increment threshold until barely left with function.
    Returns thresholded image"""
    if not (raw_image_gray_arg == None):
        raw_image_gray = raw_image_gray_arg
    else:
        raw_image_gray = cv2.cvtColor(cv2.imread(filepath),cv2.COLOR_BGR2GRAY) 
    # Now have raw_image_gray to work with (i.e. raw image in grayscale)
    # invert so line is light (higher value)
    raw_image_gray_inverted = (255-raw_image_gray)
    threshold_val = 143
    while threshold_val > 1:
    # do stuff
        print("threshold value: ", threshold_val)
        current_thresholded_image = threshold_raw(None, threshold_val,
                                    raw_image_gray_arg=raw_image_gray_inverted)
        ezshow("current thresholded image", current_thresholded_image) # show image,         
        # prompt user for input
        user_input = str(raw_input("Is the line continuous? y/n: "))
        if user_input.lower() == 'y':
        # if user input is affirmative, then break and use start_threshold-1 
            good_threshold = threshold_val + 1
            break 
        else:  
            threshold_val += -2
    # now do something with start_threshold-1 (right after user input stops it)
    final_thresholded_image = threshold_raw(None, threshold_val,
                                        raw_image_gray_arg=raw_image_gray_inverted) 
    return final_thresholded_image, threshold_val

def skeletonize(binary_image):
    """""" 
    size = np.size(binary_image)
    skeleton = np.zeros(binary_image.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    done = False

    while not done:
        eroded = cv2.erode(binary_image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_image = eroded.copy()

        zeros = size - cv2.countNonZero(binary_image)
        if zeros==size:
            done = True

    skeleton_denoised = remove_isolated_pixels(skeleton)
    return skeleton_denoised

def remove_isolated_pixels(image):
    """https://stackoverflow.com/questions/46143800/removing-isolated-pixels-using-opencv"""
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity,
        cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] < 2:
            new_image[labels == label] = 0

    return new_image

if __name__ == "__main__" and len(sys.argv) == 2:
    argv_filepath = sys.argv[1]
    actual_filepath = "%s" % argv_filepath
    raw_image_gray = cv2.imread(actual_filepath, 0)  # create bw image to use
    ROI, defined_corner_vals = user_defined_rectangle(None,
    raw_image=raw_image_gray)
    
    cropped_image_to_threshold = raw_image_gray[ROI[1]:ROI[1]+ROI[3],
    ROI[0]:ROI[0]+ROI[2]]
    binary_image, threshold_value = user_defined_threshold(None, cropped_image_to_threshold)
    skeleton = skeletonize(binary_image)
    ezshow('final skeletonized', skeleton)
    cv2.imwrite('skeletonized.png', skeleton)
