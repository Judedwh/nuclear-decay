# nuclear-decay
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
