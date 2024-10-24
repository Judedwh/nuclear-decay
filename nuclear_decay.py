#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHYS20101 Final assignment: Nuclear Decay

This programm allows for the user to calculate the values for decay constants for
Rubidium (Rb) and Strontium (Sr) as well as their respective half lives.

The program does so by reading data to arrays from input files combining them and
then removing any unwanted entries such as NaNs, errors, negative numbers
(not needed for this data) and zeroes in 'Activity Unc' column. The program also
validates the input files exist in the correct place as well as ensuring they are
of the correct form (3 columns) such that the program can run. If these requirements
are not met the user is notifed and the program halts.

The program then fits a function given by equation (4) by varying the value of
the two decay constants using a built in fmin function in order to gain a
minimised chi-squared value. Multiple fits are done until the number of outliers
in accordance with the latest fit is zero, outliers being points atleast 3 * their
uncertainty away from the fit.

After the best fitting decay constant parameters are found the uncertainties in
these values are calculated via data from a minimum chi-squared + one contour.
From these values the half lives for both Rb and Sr and their errors (via appropriate
error propogation) are calculated. Lastly based on a user input time the program
calculates the expected level of activity of Rb as well as (again via appropriate
error propogation) the error in this value. The value of the reduced chi-squared
is also calculated.

The program also produces three figures to aid in the representation
of both the data and analysis of the data carried out by the program. All of the
appropriate values (apart from the values based on user input - can be found in the
console) are displayed in the main data figure.


@author: c50392jw 12/12/2022

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.optimize import fmin
from scipy.stats import norm


def correct_column_number(input_array):
    """
    Function to check the input file has 3 columns like that of the provided
    data. Function returns True for correct form, and False for incorrect form.

    Parameters
    ----------
    input_array : Array of float64
        Array from the input file to be checked

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
        Name of intended input file

    Returns
    -------
    data : Array of float64
        Numpy array of data from input file

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
    data_array_1 : Array of float64
        Unfiltered numpy array from first input file.
    data_array_2 : Array of float64
        Unfiltered numpy array from second input file.

    Returns
    -------
    data_filtered_and_sorted : Array of float64
        Data from combined array that has been filtered and sorted

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
    time_data : Array of float64
        Single row array of first column of the input file containing time data
    decay_constant_rb : float64
        Decay constant for rubidium.
    decay_constant_sr : float64
        Decay constant for strotium.

    Returns
    -------
    expected_activity_tbq : Array of float64
        Single row array of expected activity

    '''
    time_data = 3600 * time_data
    starting_sr = constants.N_A * 10**-6
    combined_constants = starting_sr * decay_constant_rb * \
        ((decay_constant_sr) / (decay_constant_rb - decay_constant_sr))
    expected_activity_bq = combined_constants * \
        (np.exp(- decay_constant_sr * time_data) -
         np.exp(- decay_constant_rb * time_data))
    expected_activity_tbq = expected_activity_bq / 10**12
    return expected_activity_tbq


def chi_square(decay_constants_array, time_data, observed_activity, observed_activity_uncertainty):
    '''
    Function that calculates the chi_square value of the data against equation (4).
    Funciton calls the expected_activity function for the modeled data according
    to equation (4). The parameters are in such a way that they are correctly ordered
    to be then read into the scipy fmin funciton later on.

    Parameters
    ----------
    observed_activity : Array of float64
        Single row array of second column of the input file containing activity.
    expected_activity : Array of float64
        Single row array of expected activity.
    observed_activity_uncertainty : Array of float64
        Single row array of third column of the input file containing activity uncertaities.

    Returns
    -------
    chi-squared : float64
        Chi-squared value.

    '''
    decay_constant_rb = decay_constants_array[0]
    decay_constant_sr = decay_constants_array[1]
    expected_activity_data = expected_activity(
        time_data, decay_constant_rb, decay_constant_sr)

    chi_squared = np.sum((observed_activity - expected_activity_data)
                         ** 2 / (observed_activity_uncertainty)**2)
    return chi_squared


def remove_outliers(data, observed_activity, expected_activity_data, observed_activity_uncertainty):
    '''
    Function that removes outliers from combined data. Outliers are determined
    to be data points a distance atleast 3 times the activity uncertainty from
    the expected activity. Function removes outliers from main data that can be
    used further afterwards but also returns the outlying data so it can be
    plotted.

    Parameters
    ----------
    data : Array of float64
        Array of data from input file
    observed_activity : Array of float64
        Single row array of second column of the input file containing activity.
    expected_activity : Array of float64
        Single row array of expected activity.
    observed_activity_uncertainty : Array of float64
        Single row array of third column of the input file containing activity uncertaities.

    Returns
    -------
    data_complete_filter : Array of float64
        Completely filtered data array with 3 columns of time, acivity, acivity uncertainty
    outliers : Array of float64
        Outlier data array with 3 columns of time, activity, acivity uncertainty

    '''
    data_complete_filter = data[abs(
        observed_activity - expected_activity_data) < 3 * observed_activity_uncertainty]
    outliers = data[abs(
        observed_activity - expected_activity_data) >= 3 * observed_activity_uncertainty]
    number_of_outliers = len(outliers[:, 0])
    return data_complete_filter, outliers, number_of_outliers


def mesh_arrays(x_array, y_array):
    '''
    Function that creates mesh arrays for given input numpy arrays. To be then used
    in creating chi-square mesh array and then in contour plots.

    Parameters
    ----------
    x_array : Array of float64
        Input numpy array to be meshed.
    y_array : Array of float64
        Input numpy array to be meshed.

        x_array and y_array have equal length.
    Returns
    -------
    x_array_mesh : Array of float64
        Mesh array with dimensions len(x_array) X len(y_array).
    y_array_mesh : Array of float64
        Mesh array with dimensions len(x_array) X len(y_array).

    '''
    x_array_mesh = np.empty((0, len(x_array)))

    # PyLint accepts dummy_anything as an uncalled variable.
    for dummy_element in y_array:
        x_array_mesh = np.vstack((x_array_mesh, x_array))

    y_array_mesh = np.empty((0, len(y_array)))

    for dummy_element in x_array:
        y_array_mesh = np.vstack((y_array_mesh, y_array))

    y_array_mesh = np.transpose(y_array_mesh)

    return x_array_mesh, y_array_mesh


def chi_square_mesh(decay_constant_rb, decay_constant_sr, data):
    '''
    Function that creates mesh arrays for both the decay constants, then creates
    a mesh array for the chi-square values for different decay constant combinations.

    Parameters
    ----------
    decay_constant_rb_final : float64
        Calculated decay constant for rb.
    decay_constant_sr_final : float64
        Calculated decay constant for sr.
    data : Array of float64
        Filtered data from input files.

    Returns
    -------
    decay_constant_rb_mesh : Array of float64
        Mesh array of rb constants
    decay_constant_sr_mesh : Array of float64
        Mesh array of sr constants
    chi_square_values_mesh : Array of float64
        Mesh array for chi-squared values for combinations of rb and sr decay constants.
        Dimensions of mesh array are len(decay_constant_rb_mesh) X len(decay_constant_sr_mesh)

    '''
    decay_constant_rb_range = np.linspace(
        decay_constant_rb-0.03*decay_constant_rb, decay_constant_rb+0.03*decay_constant_rb, 50)
    decay_constant_sr_range = np.linspace(
        decay_constant_sr-0.03*decay_constant_sr, decay_constant_sr+0.03*decay_constant_sr, 50)

    decay_constant_rb_mesh, decay_constant_sr_mesh = mesh_arrays(
        decay_constant_rb_range, decay_constant_sr_range)

    chi_square_values_mesh = np.ones_like(decay_constant_rb_mesh)
    for row in range(len(chi_square_values_mesh)):
        for column in range(len(chi_square_values_mesh)):
            chi_square_values_mesh[row, column] = chi_square(
                (decay_constant_rb_mesh[row, column],
                 decay_constant_sr_mesh[row, column]),
                data[:, 0], data[:, 1], data[:, 2])

    return decay_constant_rb_mesh, decay_constant_sr_mesh, chi_square_values_mesh


def contour_plot(all_mesh_array, min_chi_square, decay_constant_rb, decay_constant_sr):
    '''
    Function to create a contour plot that visualises the chi-squared values for
    different decay constant values. Plot also allows for uncertainties in decay
    constants to be easily identified via the plot of horizontal and vertical lines.


    Parameters
    ----------
    mesh_array_data : tuple
        Tuple containing each of mesh arrays for the decay constants and resulting
        chi-squared mesh array. All of which are Arrays of float64.

    min_chi_square : float64
        Minimimum chi_squared value.
    decay_constant_rb : float64
        Decay constant for Rb.
    decay_constant_sr : float64
        Decay constant for Sr.

    Returns
    -------
    min_chi_square_plus_one_for_uncertainties : Array of float64
        Array for the combinations of decay constants that give min_chi_square
        + one. Array to be used later to calculate the uncertainties in the
        decay constants.

    '''
    decay_constant_rb_mesh = all_mesh_array[0]
    decay_constant_sr_mesh = all_mesh_array[1]
    chi_square_values_mesh = all_mesh_array[2]
    chi_square_array = [min_chi_square+1.00,
                        min_chi_square+2.30, min_chi_square+5.99, min_chi_square+9.21]
    contour_plt = plt.contour(decay_constant_rb_mesh, decay_constant_sr_mesh,
                              chi_square_values_mesh, chi_square_array, colors='r',
                              linestyles=('solid', 'dashed', 'dashdot', 'dotted'))
    plt.scatter(decay_constant_rb, decay_constant_sr,
                marker='+', color='k', label=r'$\chi^2_{min}$')
    plt.xlabel(r'$\lambda_{Rb}$ Values / $s^{-1}$')
    plt.ylabel(r'$\lambda_{Sr}$ Values / $s^{-1}$')
    plt.title(r'$\chi^2$ Contour Plot')
    plt.clabel(contour_plt, colors='k')
    min_chi_square_plus_one_for_uncertainties = contour_plt.allsegs[0][0]

    min_rb = np.min(min_chi_square_plus_one_for_uncertainties[:, 0])
    max_rb = np.max(min_chi_square_plus_one_for_uncertainties[:, 0])
    min_sr = np.min(min_chi_square_plus_one_for_uncertainties[:, 1])
    max_sr = np.max(min_chi_square_plus_one_for_uncertainties[:, 1])

    plt.axhline(min_sr, alpha=0.3, color='k')
    plt.axhline(max_sr, alpha=0.3, color='k')
    plt.axvline(min_rb, alpha=0.3, color='k')
    plt.axvline(max_rb, alpha=0.3, color='k')

    labels = [r'$\chi^2_{min}$+1', r'$\chi^2_{min}$+2.30',
              r'$\chi^2_{min}$+5.99', r'$\chi^2_{min}$+9.21']
    for i in range(len(labels)):
        contour_plt.collections[i].set_label(labels[i])

    plt.legend(loc='upper left', prop={'size': 9})
    plt.savefig('contour_plot_for_varied_decay_constants.png', dpi=300,
                bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    return min_chi_square_plus_one_for_uncertainties


def calculate_decay_constant_uncertainties(data_for_uncertainties):
    '''
    Function that calculates the uncertainties on the decay constants from data
    for the minimum chi-square + one contour data. Function finds the minimum and
    maximum of appropriate arrays of decay constants that result in the chi_square
    + one contour and average them to give the uncertainty.

    Parameters
    ----------
    data_for_uncertainties : Array of float64
        Data from the minimum chi-square + one contour.

    Returns
    -------
    rb_uncertainty : float64
        Value for the calculated uncertainty of Rb
    sr_uncertainty : float64
        Value for the calculated uncertainty of Sr

    '''
    rb_uncertainty = np.abs(
        np.amax(data_for_uncertainties[:, 0]) - np.amin(data_for_uncertainties[:, 0]))/2
    sr_uncertainty = np.abs(
        np.amax(data_for_uncertainties[:, 1]) - np.amin(data_for_uncertainties[:, 1]))/2
    return rb_uncertainty, sr_uncertainty


def calculate_half_lives_and_uncertainties(decay_constant_rb, decay_constant_sr,
                                           rb_uncertainty, sr_uncertainty):
    '''
    Function to calculate the half lifes for Rb and Sr as well as their respective
    uncertainties.

    Parameters
    ----------
    decay_constant_rb : float64
        Rb decay constant.
    decay_constant_sr : float64
        Sr decay constant.
    rb_uncertainty : float
        Uncertainty in Rb decay constant.
    sr_uncertainty : float
        Uncertainty in Sr decay constant.


    Returns
    -------
    half_life_rb : float64
        Value for the calculated half life of Rb.
    half_life_sr : float64
        Value for the calculated half life of Sr.
    half_life_rb_unc : float64
        Value for the calculated uncertainty of the half life of Rb.
    half_life_sr_unc : float64
        Value for the calculated uncertainty of the half life of Sr.

    '''
    half_life_rb = np.log(2) / decay_constant_rb
    half_life_sr = np.log(2) / decay_constant_sr
    half_life_rb_unc = (np.log(2) / decay_constant_rb**2) * rb_uncertainty
    half_life_sr_unc = (np.log(2) / decay_constant_sr**2) * sr_uncertainty

    return half_life_rb, half_life_sr, half_life_rb_unc, half_life_sr_unc


def predicted_activity_at_given_time(decay_constant_rb, decay_constant_sr,
                                     decay_constant_rb_unc, decay_constant_sr_unc):
    '''
    Function that asks user for a time in minutes and returns the expected activity
    according to the expected activity funciton with previously calculated best
    fitting parameters as well as uncertainties calculated using error propogation.
    Error propogated as such:
    Sigma_Avtivity^2 = (dA/dRb)^2*sigma_Rb^2 + (dA/dSr)^2*sigma_Sr^2. Apologies for
    lines 481-496 for error propogation, in line with keeping styl guide variable
    names I have created a monster.

    Parameters
    ----------
    decay_constant_rb : float64
        Rb decay constant.
    decay_constant_sr : float64
        Sr decay constant.
    decay_constant_rb_unc : float
        Uncertainty in Rb decay constant.
    decay_constant_sr_unc : float
        Uncertainty in Sr decay constant.


    Returns
    -------
    expected_activity_point : float64
        Expected activity at user input time according to expected activity function.
    time_of_interest: float64
        User input time of interest.
    expected_activity_point_unc_tbq: float64
        Ucertainty in the expected activity at user input time.


    '''
    while True:
        try:
            time_of_interest = float(input(
                "Please enter the time in minutes at which the predicted activity "
                "level is required : "))
        except ValueError:
            print("Please enter a valid time")
            continue
        if time_of_interest < 0:
            print("Please enter a non negative time")
            continue
        if time_of_interest == 0:
            print(
                "Please enter a time greater than 0")
        else:
            break

    time_in_hours = time_of_interest / 60
    expected_activity_point = expected_activity(
        time_in_hours, decay_constant_rb, decay_constant_sr)

    # Error propogation on expected activity at time of interest

    time_in_secs = time_of_interest*60
    starting_sr = constants.N_A * 10**-6

    activity_by_rb = (-starting_sr*decay_constant_sr *
                      (decay_constant_sr*np.exp(time_in_secs*decay_constant_rb)
                       - time_in_secs*np.exp(time_in_secs*decay_constant_sr) *
                       decay_constant_rb**2+time_in_secs*decay_constant_sr *
                       np.exp(time_in_secs*decay_constant_sr) *
                       decay_constant_rb
                       - decay_constant_sr*np.exp(time_in_secs*decay_constant_sr))
                      * np.exp(-time_in_secs*(decay_constant_rb+decay_constant_sr))
                      )/(decay_constant_rb-decay_constant_sr)**2
    activity_by_sr = (-starting_sr*decay_constant_rb *
                      (decay_constant_rb*np.exp(time_in_secs*decay_constant_sr)
                       - time_in_secs*np.exp(time_in_secs*decay_constant_rb) *
                       decay_constant_sr**2+time_in_secs*decay_constant_rb *
                       np.exp(time_in_secs*decay_constant_rb)*decay_constant_sr
                       - decay_constant_rb*np.exp(time_in_secs*decay_constant_rb))
                      * np.exp(-time_in_secs*(decay_constant_rb+decay_constant_sr))
                      )/(decay_constant_sr-decay_constant_rb)**2
    expected_activity_point_unc = np.sqrt(
        activity_by_rb**2*decay_constant_rb_unc**2+activity_by_sr**2*decay_constant_sr_unc**2)
    expected_activity_point_unc_tbq = expected_activity_point_unc / 10**12
    return expected_activity_point, time_of_interest, expected_activity_point_unc_tbq


def plot_data(data, expected_data, data_outliers, decay_constant_values,
              half_life_values, reduced_chi_squared):
    '''
    Function to plot all of the raw data with error bars including the removed
    data points based on outlier analysis and fitted function with calculated best
    fit parameters. Function also annotates the graph with calculated values.

    Returns
    -------
    None.

    '''

    time_data = 3600*data[:, 0]
    activity_data = data[:, 1]
    activity_data_unc = data[:, 2]

    fig = plt.figure()
    axes_data = fig.add_subplot(4, 1, (1, 3))
    axes_data.plot(time_data, expected_data,
                   label='Expected Activity Fit', color='red', linestyle='-')
    axes_data.plot(3600*data_outliers[:, 0], data_outliers[:, 1], linestyle='none',
                   marker='*', color='r', label='Removed Data Points')
    axes_data.errorbar(time_data, activity_data, activity_data_unc, linestyle="none",
                       label='Data', alpha=0.5, fmt='+k',
                       markersize='4')
    axes_data.legend(loc='best', prop={'size': 8})
    axes_data.set_xlabel("Time / s", fontname='Arial',
                         fontsize='12')
    axes_data.set_ylabel(r"Activity / $TBq$", fontname='Arial', fontsize='12',)
    axes_data.set_title('Activity of ${}^{79}Rb$ against Time',
                        fontname='Arial', fontsize='14')
    axes_data.grid(dashes=[4, 2], linewidth=1.2)

    axes_data.annotate((r'$\lambda_{{Rb}}$ = {0:.6f} ± {1:.6f}$s^{{-1}}$'
                        .format(decay_constant_values[0], decay_constant_values[2])),
                       (0, 0), (25, -55), xycoords='axes fraction', va='top',
                       textcoords='offset points', fontsize='10')
    axes_data.annotate((r'$\lambda_{{Sr}}$ = {0:.3g} ± {1:.5f}$s^{{-1}}$'
                        .format(decay_constant_values[1], decay_constant_values[3])),
                       (0, 0), (25, -70), xycoords='axes fraction', va='top',
                       textcoords='offset points', fontsize='10')
    axes_data.annotate(('Rb $t_{{1/2}}$ = {0:.3g} ± {1: .1f}mins'
                        .format(half_life_values[0]/60, half_life_values[2]/60)),
                       (0, 0), (200, -55), xycoords='axes fraction',
                       textcoords='offset points', va='top', fontsize='10')
    axes_data.annotate(('Sr $t_{{1/2}}$ = {0:.3g} ± {1: .2f}mins'
                        .format(half_life_values[1]/60, half_life_values[3]/60)),
                       (0, 0), (200, -70), xycoords='axes fraction',
                       textcoords='offset points', va='top', fontsize='10')
    axes_data.annotate((r'$\chi^2$ Reduced = {0:.2f}'.format(reduced_chi_squared)), (0, 0),
                       (125, -90), xycoords='axes fraction',
                       textcoords='offset points', va='top', fontsize='10')
    plt.savefig('activity_of_rb_against_time.png', dpi=300,
                bbox_inches='tight')

    plt.show()


def residuals_plots(data, expected_data):
    '''
    Function to plot the residuals of the data against the fit function. The
    raw residuals are plot as well as a histogram to show the spread of residuals.
    A probability density function fit is also plotted over the histogram to show
    a true gaussian distribution for the residual data.

    Returns
    -------
    None.

    '''

    time_data = data[:, 0]
    activity_data = data[:, 1]
    activity_data_unc = data[:, 2]

    fig = plt.figure()
    residuals = activity_data - expected_data
    axes_residuals = fig.add_subplot(5, 1, (1, 2))
    axes_residuals.errorbar(time_data, residuals, yerr=activity_data_unc,
                            markersize='2', fmt='o', color='k')
    axes_residuals.plot(time_data, 0 * time_data, color='red')
    axes_residuals.grid(True)
    axes_residuals.set_title('Residuals', fontsize=14, fontname='Arial')
    axes_residuals.set_ylabel(
        r"Residual / TBq", fontname='Arial', fontsize='12',)

    residual_mean, residual_std = norm.fit(residuals)

    # Plot the histogram.
    axes_residuals_hist = fig.add_subplot(5, 1, (4, 5))
    axes_residuals_hist.hist(
        residuals, bins=15, density=True, alpha=0.4, color='k')
    # Plot the PDF.
    residual_min, residual_max = plt.xlim()
    residuals = np.linspace(residual_min, residual_max, 100)
    normal_residuals = norm.pdf(residuals, residual_mean, residual_std)
    axes_residuals_hist.plot(residuals, normal_residuals, 'r-.',
                             linewidth=2, label="Residual Normal Distribution")
    axes_residuals_hist.legend(loc='best', prop={'size': 8})
    axes_residuals_hist.set_title(
        "Residual histogram with residual normal distribution ", fontsize=14, fontname='Arial')
    axes_residuals_hist.set_xlabel("Residual / TBq", fontname='Arial',
                                   fontsize='12')
    axes_residuals_hist.set_ylabel(
        r"Probability", fontname='Arial', fontsize='12',)
    plt.savefig('residuals_plots.png', dpi=300,
                bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def main():
    '''
    Function to run whole program. While loop included such that trial fits are
    applied to the data until the number of outliers (in accordance with remove_outliers
    function) is equal to 0. For the given data, this means that the while loop
    is only passed through once as all of the three outliers in the data are removed
    in the first iteration through the while loop. The hope in including this while
    loop is to make the program more general such that it would be able to deal
    with datasets with less obvious outliers which would be caught by continously
    fitting better fits hence revealing previosly fine data points as outliers.



    Returns
    -------
    int.

    '''
    data_array_1 = read_data_to_array('nuclear_data_1.csv')
    data_array_2 = read_data_to_array('nuclear_data_2.csv')
    stacked_array = combine_and_filter_data_arrays(data_array_1, data_array_2)
    total_outliers = np.zeros((0, 3))

    decay_constant_rb = 0.0005
    decay_constant_sr = 0.005
    decay_constants_array = [decay_constant_rb, decay_constant_sr]
    first_trial = True
    number_of_outliers = 0

    while number_of_outliers > 0 or first_trial:
        first_trial = False
        minimised_chi_square = fmin(chi_square, decay_constants_array,
                                    full_output=True, disp=False, args=(
                                        stacked_array[:,
                                                      0], stacked_array[:, 1],
                                        stacked_array[:, 2]))
        decay_constant_rb, decay_constant_sr = minimised_chi_square[0]
        decay_constants_array = [decay_constant_rb, decay_constant_sr]
        expected_activity_data = expected_activity(
            stacked_array[:, 0], decay_constant_rb, decay_constant_sr)
        stacked_array, outliers_data, number_of_outliers = remove_outliers(stacked_array,
                                                                           stacked_array[:, 1],
                                                                           expected_activity_data,
                                                                           stacked_array[:, 2])
        total_outliers = np.vstack((total_outliers, outliers_data))

    data_final = stacked_array
    decay_constant_rb_final = minimised_chi_square[0][0]
    decay_constant_sr_final = minimised_chi_square[0][1]
    reduced_chi_squared = minimised_chi_square[1] / (len(data_final) - 2)
    all_mesh_arrays = chi_square_mesh(
        decay_constant_rb_final, decay_constant_sr_final, data_final)
    data_for_uncertainties = contour_plot(all_mesh_arrays, minimised_chi_square[1],
                                          decay_constant_rb_final, decay_constant_sr_final)
    rb_uncertainty, sr_uncertainty = calculate_decay_constant_uncertainties(
        data_for_uncertainties)

    half_life_values = calculate_half_lives_and_uncertainties(
        decay_constant_rb, decay_constant_sr, rb_uncertainty, sr_uncertainty)

    decay_constant_values = [decay_constant_rb_final,
                             decay_constant_sr_final, rb_uncertainty, sr_uncertainty]

    prediction_values = predicted_activity_at_given_time(
        decay_constant_rb_final, decay_constant_sr_final, rb_uncertainty, sr_uncertainty)

    plot_data(data_final, expected_activity_data, total_outliers,
              decay_constant_values, half_life_values, reduced_chi_squared)
    residuals_plots(data_final, expected_activity_data)
    print("The best-fitting value for the Rb decay constant is {0:.6f} ± {1:.6f}/s".format(
        decay_constant_rb_final, rb_uncertainty))
    print("The best-fitting value for the Sr decay constant is {0:.3g} ± {1:.5f}/s".format(
        decay_constant_sr_final, sr_uncertainty))
    print("The reduced chi-squared is {0:.2f}".format(reduced_chi_squared))
    print(
        "The half-life for Rb is {0:.3g} ± {1:.1f}mins ".format(half_life_values[0]/60,
                                                                half_life_values[2]/60))
    print(
        "The half-life for Sr is {0:.3g} ± {1:.2f}mins ".format(half_life_values[1]/60,
                                                                half_life_values[3]/60))
    print("The predicted activity at time = {0:}mins is {1:.3g} ± {2:.1f}TBq ".format(
        prediction_values[1], prediction_values[0], prediction_values[2]))

    return 0


if __name__ == "__main__":
    main()
