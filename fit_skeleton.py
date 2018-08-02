"""TODO: ensure that skeleton image is in fact binary"""
import numpy as np
import math
from pynverse import inversefunc
import cv2
from plot_simple_frequency import *

def find_LCI(binary_image, corner_values, Lgeo_range, Cgeo_range, Ic_range):
    Lgeo_numpoints = len(Lgeo_range)
    Cgeo_numpoints = len(Cgeo_range)
    Ic_numpoints = len(Ic_range)
    
    a1 = corner_values[0]
    a2 = corner_values[1]
    y1 = corner_values[2]
    y2 = corner_values[3]
    
    LCI_scores = np.empty((Lgeo_numpoints, Cgeo_numpoints, Ic_numpoints)) 

    for L_index in range(Lgeo_numpoints):
        L = Lgeo_range[L_index]

        for C_index in range(Cgeo_numpoints):
            C = Cgeo_range[C_index]

            for I_index in range(Ic_numpoints):
                I = Ic_range[I_index]
                
                # simple_plot_frequency(L, C, I, a1, a2, y1, y2)
                print("%d %d %d" % (L_index, C_index, I_index)) 
                # Now calculate score (call get_square_score function)
                score = get_square_score(binary_image, corner_values, L, C,
                I)
                LCI_scores[L_index, C_index, I_index] = score
    
    min_indices = np.where(LCI_scores==LCI_scores.min())
    Lgood = Lgeo_range[min_indices[0]]
    Cgood = Cgeo_range[min_indices[1]]
    Icgood = Ic_range[min_indices[2]]
    return Lgood, Cgood, Icgood
     
    ### eventually plot directly against 
def get_square_score(matrix, corner_vals, L, C, I):
    matshape = np.shape(matrix)
    num_rows = matshape[0]
    num_cols = matshape[1]

    f1 = float(corner_vals[0])
    f2 = float(corner_vals[1])
    freq1 = float(corner_vals[2])
    freq2 = float(corner_vals[3])
    
    total_score = 0

    for col in range(num_cols):
        fDC_val = f1 + (f2-f1)*(col/float(num_cols))
        this_func_val = get_function_value(fDC_val, L, C, I)
        # for point in column:
        this_column = matrix[:, col]
        pre_nonzero_points = cv2.findNonZero(this_column)
        if pre_nonzero_points is None:
            total_score += 0
        else:
            nonzero_points = pre_nonzero_points.astype(np.uint8)[:, 0, :]
            #print('nonzero points', nonzero_points)
            #print('shape of nonzero_points', np.shape(nonzero_points))
            vert_pos_total_for_avg = 0
            for i in range(len(nonzero_points)):
                vert_pos = nonzero_points[i,1]
                vert_pos_total_for_avg += vert_pos
            vert_pos_avg = vert_pos_total_for_avg/float(i+1) 
            freq_val = freq2-(vert_pos_avg/float(num_rows))*(freq2-freq1)
            
            squared_dist = (this_func_val-freq_val)**2
            total_score += squared_dist
    ##print("total score")
    #print(total_score)
    return total_score

def get_function_value(fDC, L, C, Ic):
    """"""
    fDC = float(fDC)
    L = float(L)
    C = float(C)
    Ic = float(Ic)      # just making sure types aren't wrong somehow
    beta_RF = 2*math.pi*L*Ic/(2.068*1.e-15) 
    # get dDC from fDC:
    fDC_func = (lambda x: (x + beta_RF*math.sin(x))/(2*math.pi))
    fDC_inv = inversefunc(fDC_func)  # function that goes from fDC to dDC
    dDC = fDC_inv(fDC)  # use that inverted function to get dDC value

    Ljj = (2.068*1.e-15)/(2*math.pi*Ic*math.cos(dDC))
    Ltot = (1/L+1/Ljj)**(-1)    # parallel inductances
    try:
        resonant_frequency = (1/(2*math.pi)) * (Ltot*C)**(-.5)
    except:
        resonant_frequency = 0
    return np.real(resonant_frequency)
