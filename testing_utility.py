import numpy as np
import scipy
from scipy import sparse
import time
import open3d as o3d

import struct

def magnitude_filter(sample, floor, ceiling):
    # This is horrible and very, very ugly but we will have to bear with it. (prepare a bucket)
    # floor: the lowest order of magnitude, must be int (it will be rounded by typecasting)
    # ceiling: the highest order of magnitude, must be int (it will be rounded by typecasting)
    census = []             # This will contain the number of elements in sample that fall into each zone, between orders of magnitude
    catalogue = []          # This will contain arrays of elements in sample that fall into each zone, between orders of magnitude
    index_catalogue = []    # This is a stillbirth between this and the previous version. Will be fixed in a subsequent edition. Like catalogue, but with indices.
    
    # Initialising O.O.M. zones
    orders = np.linspace(int(floor), int(ceiling), num=int(ceiling-floor+1), dtype=int)
    
    catalogue.append(sample[np.where(sample<10.0**(orders[0]))])
    index_catalogue.append(np.where(sample < 10.0**(orders[0]))[0])
    census.append(len(catalogue[-1]))
    
    for i in range(0, len(orders)-1):
        #where = np.where(sample < 10.0**(orders[i]))[0] This can work.
        a = sample >= 10.0**(orders[i])
        b = sample < 10.0**(orders[i+1])
        
        temp = []
        indices_temp = []
        for j in range(0, len(sample)):
            if a[j] == b[j]:
                temp.append(sample[j])
                indices_temp.append(j)
        catalogue.append(np.array(temp))
        index_catalogue.append(np.array(indices_temp))
        census.append(len(catalogue[-1]))
    
    catalogue.append(sample[np.where(sample>=10.0**(orders[-1]))])
    index_catalogue.append(np.where(sample >= 10.0**(orders[-1]))[0])
    census.append(len(catalogue[-1]))
    # The reader was warned.
    
    # More Debugging
    ###buffer = np.copy(sample)
    ###exceptions = []
    ###for i in range(0, len(buffer)):
    ###    disabler = 0
    ###    for j in range(0, len(catalogue)):
    ###        if buffer[i] in catalogue[j]:
    ###            disabler = 1
    ###            break
    ###    if disabler == 0:
    ###        exceptions.append([i,buffer[i]])
    ###print(exceptions)
    
    return np.array(census), catalogue, index_catalogue
    
def halfway_filter(floor, ceiling, sample):
    # Returns arrays, containing each element of sample that is above or below the midpoint of floor and ceiling.
    # If sample[i] equals the midpoint, it is considered greater than it.
    divider = (ceiling - floor)/2
    where_above = np.where(sample>=divider)
    
    above = sample[where_above]
    below = sample[np.where(sample<divider)]
    
    return above, below, where_above[0]
    
def pt_of_interest_halfway(pt_index, floor, ceiling, sample_order):
    # Returns the results of halfway_filter for a certain index of a magnitude filter, previously applied on a sample array. More tidy than writing the code in main.py. 
    # pt_index: the given index of interest (usually the most densely populated one).
    # floor, ceiling: the parameters passed to the magnitude filter.
    # sample_order: an array of the elements populating pt_index
    bar_upper = 10.0**(floor+pt_index)
    bar_lower = 10.0**(floor+pt_index-1)
    
    above, below, where_above = halfway_filter(bar_lower, bar_upper, sample_order)
    
    return above, below, where_above