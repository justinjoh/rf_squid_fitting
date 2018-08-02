import numpy as np
import opencv as cv2
import sys

from fit_skeleton import *
from skeletonize_image import *
from plot_simple_frequency import *

Lgeo_numpoints = 10
Cgeo_numpoints = 10
Ic_numpoints = 10
Lgeo_range = np.linspace(4*1e-12, 6*1e-12, Lgeo_numpoints)
Cgeo_range = np.linspace(50*1e-12, 65*1e-12, Cgeo_numpoints)
Ic_range = np.linspace(20*1e-6, 50*1e-6, Ic_numpoints)

"""
Lgeo_numpoints = 1
Cgeo_numpoints = 1
Ic_numpoints = 1
Lgeo_range = np.linspace(19/2*1e-12, 19*1e-12, 1)
Cgeo_range = np.linspace(30*1e-12, 30*1e-12, 1)
Ic_range = np.linspace(23*1e-6, 23*1e-6, 1)
"""

if __name__ == "__main__" and (len(sys.argv)==2):
    argv_filepath = sys.argv[1]
    actual_filepath = "%s" % argv_filepath  # why did I do it this way?
    raw_image_gray = cv2.imread(actual_filepath, 0)
    ROI, corner_vals = user_defined_rectangle(None,
        raw_image=raw_image_gray)
    
    x1 = float(corner_vals[0])
    x2 = float(corner_vals[1])
    y1 = float(corner_vals[2])
    y2 = float(corner_vals[3])

    cropped_image_to_threshold = raw_image_gray[ROI[1]:ROI[1]+ROI[3],
        ROI[0]:ROI[0]+ROI[2]]
    binary_image, threshold_value = user_defined_threshold(None,
        cropped_image_to_threshold)
    
    # skeletonize here
    skeleton = skeletonize(binary_image)

    # now find good parameters
    L, C, I = find_LCI(skeleton, corner_vals, Lgeo_range, Cgeo_range, Ic_range)
    print("L ", float(L))
    print("C ", float(C))
    print("I ", float(I))
    simple_plot_frequency_on_image(skeleton, L, C, I, x1, x2, y1, y2) 
