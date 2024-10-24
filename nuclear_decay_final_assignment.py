#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHYS20101 Final assignment: Nuclear Decay

@author: judewh
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import constants
from scipy.optimize import fmin


decay_constant_rb = 0.0005
decay_constant_sr = 0.005
decay_constants_array = [decay_constant_rb, decay_constant_sr]


def correct_column_number(input_array):
    """
    Function to check the input file has 3 columns like that of the provided 
    data. Function returns True for correct form, and False for incorrect form

    Parameters
    ----------
    input_array : numpy array
        array from the input file to be checked

    Returns
    -------
    is_three_columns : Boolean
        True when 3 columns
        False when not 3 columns

    """
    _, columns = np.shape(input_array)  # row part of shape uncalled
    if columns != 3:
        three_columns = False
    else:
        three_columns = True
    return three_columns


def read_data_to_array(file_name):
    """
    Function to first validate that the intended input file exists (in correct place)
    and then read in the input file into a numpy array. The function also calls
    the correct_column_number function to then validate the input file has 3 columns.
    If either validation attempts are not met the user will be prompted.


    Parameters
    ----------
    file_name : string
        name of intended input file

    Returns
    -------
    data : numpy array
        numpy array of data from input file

    """
    input_file = str(file_name)

    try:
        input_file = open(input_file, "r")
    except FileNotFoundError:
        print(
            'Unable to open file with name {0}\nPlease ensure the file name is'
            ' correct'.format(file_name))
        sys.exit()
    data = np.genfromtxt(input_file, delimiter=",")
    return data
    if correct_column_number(data):
        return data
    input_file.close()
    print('Input file does not contain the correct number of columns for this'
          'program, please enter input a file with 3 columns')
    sys.exit()


def combine_and_filter_data_arrays(data_array_1, data_array_2):
    '''
    Function to firtsly combine 2 input files then filter any faulty measurments 
    and lastly sort data in ascending time. NaNs are filtered as well as any negative
    numbers along with numbers = 0 in the Activity and Activity Unc columns.


    Parameters
    ----------
    data_array_1 : numpy array
        unfiltered numpy array from first input file.
    data_array_2 : numpy array
        unfiltered numpy array from second input file.

    Returns
    -------
    data_filtered_and_sorted : 
        data from combined array that has been filtered and sorted 

    '''
    stacked_data = np.vstack((data_array_1, data_array_2))
    stacked_data_nans_filtered = stacked_data[~np.isnan(
        stacked_data).any(axis=1)]
    stacked_data_negatives_filtered = stacked_data_nans_filtered[np.any(
        stacked_data_nans_filtered >= 0, axis=1)]
    stacked_data_zero_activity_filtered = stacked_data_negatives_filtered[
        stacked_data_nans_filtered[:, 1] != 0]
    stacked_data_complete_filtered = stacked_data_zero_activity_filtered[
        stacked_data_nans_filtered[:, 2] != 0]
    data_filtered_and_sorted = stacked_data_complete_filtered[np.argsort(
        stacked_data_complete_filtered[:, 0])]

    return data_filtered_and_sorted


def expected_activity(time_data, decay_constant_rb, decay_constant_sr):
    '''
    Function that takes in time data from input file and returns the expected activity
    according to equation (4). The expected activity is also converted to TBq.

    Parameters
    ----------
    time_data : numpy array
        Single row array of first column of the input file containing time data
    decay_constant_rb : float
        Decay constant for rubidium. Initial value stated used.
    decay_constant_sr : float
        Decay constant for strotium. Inital value stated used.

    Returns
    -------
    expected_activity_tbq : numpy array
        Single row array of expected activity 

    '''
    time_data = 3600 * time_data
    starting_sr = constants.N_A * 10**-6
    combined_constants = starting_sr * decay_constant_rb * \
        ((decay_constant_sr) / (decay_constant_rb - decay_constant_sr))
    expected_activity = combined_constants * \
        (np.exp(- decay_constant_sr * time_data) -
         np.exp(- decay_constant_rb * time_data))
    expected_activity_tbq = expected_activity / 10**12
    return expected_activity_tbq


def chi_square(decay_constants_array, time_data, observed_activity, observed_activity_uncertainty):
    '''
    Function that calculates the chi_square value of the data against equation (4).
    Funciton calls the expected_activity function for the modeled data according
    to equation (4). The parameters are in such a way that they are correctly ordered
    to be then read into the scipy fmin funciton later on.

    Parameters
    ----------
    observed_activity : numpy array 
        Single row array of second column of the input file containing activity.
    expected_activity : numpy array
        Single row array of expected activity.
    observed_activity_uncertainty : numpy array
        Single row array of third column of the input file containing activity uncertaities.

    Returns
    -------
    chi-squared : float
        chi-square value.

    '''
    decay_constant_rb = decay_constants_array[0]
    decay_constant_sr = decay_constants_array[1]
    expected_activity_data = expected_activity(
        time_data, decay_constant_rb, decay_constant_sr)

    chi_squared = np.sum((observed_activity - expected_activity_data)
                         ** 2 / (observed_activity_uncertainty)**2)
    return chi_squared


def remove_outliers(data, observed_activity, expected_activity, observed_activity_uncertainty):
    '''
    Function that removes outliers from combined data. Outliers are determined 
    to be data points a distance atleast 3 times the activity uncertainty from 
    the expected activity. Function removes outliers from main data that can be 
    used further afterwards but also returns the outlying data so it can be 
    plotted.

    Parameters
    ----------
    observed_activity : numpy array
        Single row array of second column of the input file containing activity.
    expected_activity : numpy array
        Single row array of expected activity.
    observed_activity_uncertainty : numpy array
        Single row array of third column of the input file containing activity uncertaities.

    Returns
    -------
    data_complete_filter : numpy array
        completely filtered data array with 3 columns of time, acivity, acivity uncertainty
    outliers : numpy array
        outlier data array with 3 columns of time, acivity, acivity uncertainty

    '''
    data_complete_filter = data[abs(
        observed_activity - expected_activity) < 3 * observed_activity_uncertainty]
    outliers = data[abs(
        observed_activity - expected_activity) >= 3 * observed_activity_uncertainty]
    return data_complete_filter, outliers


data_array_1 = read_data_to_array('nuclear_data_1.csv')

data_array_2 = read_data_to_array('nuclear_data_2.csv')

stacked_array = combine_and_filter_data_arrays(data_array_1, data_array_2)

expected_activity_data = expected_activity(
    stacked_array[:, 0], decay_constant_rb, decay_constant_sr)

data_final, data_outliers = remove_outliers(stacked_array,
                                            stacked_array[:, 1], expected_activity_data, stacked_array[:, 2])

minimised_chi_square = fmin(chi_square, decay_constants_array, full_output=True, disp=False, args=(
    data_final[:, 0], data_final[:, 1], data_final[:, 2]))

reduced_chi_square = minimised_chi_square[1] / (len(data_final) - 2)

decay_constant_rb_final = minimised_chi_square[0][0]
decay_constant_sr_final = minimised_chi_square[0][1]

# print(stacked_array)
# print(data_final)
# print(data_outliers)

# time_data = 3600 * stacked_array[:, 0]
# activity_data = stacked_array[:, 1]
# activity_unc_data = stacked_array[:, 2]


# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.set_title('Activity against Time', fontsize=14)
# ax.set_xlabel('Time / seconds', fontsize=14)
# ax.set_ylabel('Activity / TBq', fontsize=14)

# ax.tick_params(labelsize=14)

# ax.errorbar(time_data, activity_data, yerr=activity_unc_data, fmt='o')
# ax.plot(time_data, expected_activity_data)
# plt.show()

time_data = 3600 * data_final[:, 0]
activity_data = data_final[:, 1]
activity_unc_data = data_final[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('Activity against Time', fontsize=14)
ax.set_xlabel('Time / seconds', fontsize=14)
ax.set_ylabel('Activity / TBq', fontsize=14)

# ax.tick_params(labelsize=14)

ax.plot(3600 * stacked_array[:, 0], expected_activity_data,
        label='Fit function', color='red', linestyle='-')
ax.plot(3600 * data_outliers[:, 0], data_outliers[:, 1], linestyle='none',
        marker='*', color='red', label='Data outliers')
ax.errorbar(time_data, activity_data, yerr=activity_unc_data,
            fmt='+k', label='Raw Data')
# plt.savefig('Original Plot.png', dpi=600, bbox_inches='tight')
plt.show()
