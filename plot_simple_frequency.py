import matplotlib.pyplot as plt
import numpy as np
import math
from pynverse import inversefunc
import sys


def get_simple_frequencies(list_of_fDC, Lgeo, Cgeo, Ic):

    beta_RF = 2*math.pi*Lgeo*Ic/(2.068*1.e-15)

    num_fDC_points = len(list_of_fDC)
    f0_arr = np.empty((num_fDC_points, 1))  # initialize array to return
    # iterate through list_of_fDC, and calculate resonant frequency
    for i in range(0, num_fDC_points):
        fDC = list_of_fDC[i]
        f0_arr[i] = simple_resonant_frequency(Lgeo, Cgeo, Ic, beta_RF, fDC) # calculate here
    return f0_arr


def simple_resonant_frequency(L, C, Ic, beta_RF, fDC):
    # first, need to get dDC from fDC.
    fDC_func = (lambda x: (x + beta_RF*math.sin(x))/(2*math.pi))
    fDC_inv = inversefunc(fDC_func) # function that goes from fDC to dDC
    dDC = fDC_inv(fDC)  # use that inverted function to get dDC value
    # dDC = fDC*2*math.pi
    Ljj = (2.068*1.e-15)/(2*math.pi*Ic*math.cos(dDC))
    Ltot = (1/L+1/Ljj)**(-1)    # parallel inductances
    resonant_frequency = (1/(2*math.pi)) * (Ltot*C)**(-.5)
    return np.real(resonant_frequency)


if __name__ == '__main__' and sys.argv[1] == "linspace":
    """
    For using np.linspace
    """

    # ---
    fDC_start = -1.55  # start number of flux quanta
    fDC_stop = 1.55  # stop number of flux quanta
    fDC_numsteps = 100  # number of flux quantum steps

    Lgeo_start = (.66*1e-9)/2
    Lgeo_stop = (.66*1e-9)/2
    Lgeo_numpoints = 1

    Cgeo_start = .42*1e-12
    Cgeo_stop = .42*1e-12
    Cgeo_numpoints = 1
    Cgeo = 30*1e-12

    Ic_start = .55 * 1e-6
    Ic_stop = 20*1e-6
    Ic_numpoints = 1

    # ---
    Lgeo_linspace = np.linspace(Lgeo_start, Lgeo_stop, Lgeo_numpoints)
    Cgeo_linspace = np.linspace(Cgeo_start, Cgeo_stop, Cgeo_numpoints)
    Ic_linspace = np.linspace(Ic_start, Ic_stop, Ic_numpoints)

    print("Lgeo values", Lgeo_linspace)
    print("Cgeo values", Cgeo_linspace)
    print("Ic values", Ic_linspace)

    for Lgeo in Lgeo_linspace:
        for Ic in Ic_linspace:
            for Cgeo in Cgeo_linspace:
                fDC_list = np.linspace(fDC_start, fDC_stop, fDC_numsteps)  # i.e. x-axis values
                resonant_frequency_list = get_simple_frequencies(fDC_list, Lgeo, Cgeo, Ic)

                plt.plot(fDC_list, resonant_frequency_list)
                plt.xlabel('applied DC flux'); plt.ylabel('resonant frequency')

    # plt.ylim(ymin=4.5*1e9)
    # plt.ylim(ymax=13e9)
    plt.plot(fDC_list, np.ones((len(fDC_list), 1)) * 9e9)
    plt.plot(fDC_list, np.ones((len(fDC_list), 1)) * 17e9)
    plt.legend()
    plt.show()

if __name__ == '__main__' and sys.argv[1] == "list":
    """
    For running with specific set of values
    """

    fDC_start = -.6  # start number of flux quanta
    fDC_stop = .6  # stop number of flux quanta
    fDC_numsteps = 1000  # number of flux quantum steps

    Lgeo_values = [19/2*1e-12]
    Cgeo_values = [30*1e-12]
    Ic_values = [23*1e-6]

    for Lgeo in Lgeo_values:
        for Cgeo in Cgeo_values:
            for Ic in Ic_values:
                fDC_list = np.linspace(fDC_start, fDC_stop, fDC_numsteps)  # i.e. x-axis values
                resonant_frequency_list = get_simple_frequencies(fDC_list, Lgeo, Cgeo, Ic)
                plt.plot(fDC_list, resonant_frequency_list, 'g-')
                plt.xlabel('applied DC flux'); plt.ylabel('resonant frequency')


    #plt.xlim(xmin=-1), plt.xlim(xmax=1)
    plt.ylim(ymin=5*1e9); plt.ylim(ymax=13e9)
    #plt.plot(fDC_list, np.ones((len(fDC_list), 1))*17e9)
    #plt.plot(fDC_list, np.ones((len(fDC_list), 1))*10e9)
    plt.legend()
    plt.show()

    '''
    fDC_list = np.linspace(0, 20, 1000)
    to_plot = test_inverse_stuff(fDC_list)
    plt.plot(fDC_list, to_plot)
    to_plot2 = test_inverse_stuff2(fDC_list)
    plt.plot(fDC_list, to_plot2)
    plt.plot(fDC_list, fDC_list)
    plt.ylim((0, 20))
    plt.xlim((0, 20))
    plt.show()
    '''

    if __name__ == '__main__' and sys.argv[1] == "import":
        """
        python plot_simple_frequency.py import somefile.txt
        For reading in from data file
        Data file format:
        <fDC_start>
        <fDC_stop>
        <fDC_numsteps>
        "
        """
        filename = sys.argv[2]
        datafile = open(filename, 'r')  # set to read file
        readfile = datafile.readlines()
        pass





if __name__ == '__main__' and len(sys.argv) == 1:
    # ---
    fDC_start = -.7  # start number of flux quanta
    fDC_stop = .7  # stop number of flux quanta
    fDC_numsteps = 6000  # number of flux quantum steps

    Lgeo = (.33*1e-9)
    Cgeo = .42*1e-12
    Ic = .55*1e-6
    # ---

#    for Lgeo in [10*1e-12, 15*1e-12]:

    fDC_list = np.linspace(fDC_start, fDC_stop, fDC_numsteps)  # i.e. x-axis values
    resonant_frequency_list = get_simple_frequencies(fDC_list, Lgeo, Cgeo, Ic)

    plt.plot(fDC_list, resonant_frequency_list)
    plt.xlabel('applied DC flux'); plt.ylabel('resonant frequency')

    #plt.ylim(ymin=4.5*1e9)
    #plt.ylim(ymax=13e9)
    plt.plot(fDC_list, np.ones((len(fDC_list), 1))*17e9)
    plt.plot(fDC_list, np.ones((len(fDC_list), 1))*10e9)
    plt.legend()
    plt.show()

    '''
    fDC_list = np.linspace(0, 20, 1000)
    to_plot = test_inverse_stuff(fDC_list)
    plt.plot(fDC_list, to_plot)
    to_plot2 = test_inverse_stuff2(fDC_list)
    plt.plot(fDC_list, to_plot2)
    plt.plot(fDC_list, fDC_list)
    plt.ylim((0, 20))
    plt.xlim((0, 20))
    plt.show()
    '''

