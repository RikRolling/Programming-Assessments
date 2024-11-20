#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Determining the mass (m_z) and lifetime (tau) of
the Z Boson.

Ritika Khot, University ID: 10630885

26/11/21

Calculates the mass and lifetime of the z boson
from a Gaussian plot generated from two data sets
which are read in and validated, then combined
to form this plot.

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.optimize import fmin


FILE_NUMBER_1 = 'z_boson_1.csv'
FILE_NUMBER_2 = 'z_boson_2.csv'

WIDTH_EE = 83.91*10**(-3)#GeV

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_file(file_name):
    """
    Parameters
    ----------
    file_name : delimited csv file
        Reads in data file

    Returns
    -------
    input_file : delimited csv file
    """
    try:

        input_file = np.genfromtxt(file_name, delimiter=',', comments='%',\
                               skip_header=1)

    except OSError:
        sys.tracebacklimit = 0

        raise OSError('File not found. Please try again.')

    return input_file


def file_validation(file_inputted):
    """
    Parameters
    ----------
    file_inputted : delimited text file

        Removes any nans and uncertainties that equal 0 from csv file.

    Returns
    -------
    validated_array : array of floats
    """

    removal_of_nan_array = file_inputted[~np.isnan(file_inputted[:, 0]*\
                                file_inputted[:, 1]*file_inputted[:, 2])]\

    validated_array = removal_of_nan_array[removal_of_nan_array[:, 2] != 0]

    return validated_array


def array_combined(array_1, array_2):
    """
    Parameters
    ----------
    array_1 : array of floats
        Data from FILE_NUMBER_1.
    array_2 : array of floats
        Data from FILE_NUMBER_2.

    Returns
    -------
    combined_array : array of floats
        Combined data from FILE_NUMBER_1 and FILE_NUMBER_2.
    """

    combined_array = np.vstack((array_1, array_2))

    return combined_array



def removing_outliers(combined_data):
    """
    Parameters
    ----------
    combined_data : array


    Returns
    -------
    array

    combined_data

        Data outside of +/- 3*sigma level are removed

    """

    mean = np.mean(combined_data[:, 1])
    sigma = np.std(combined_data[:, 1])
    combined_data = combined_data[(combined_data[:, 1] > mean - 3*sigma)*\
                                  (combined_data[:, 1] < mean+3*sigma)]

    return combined_data


def cross_section_function(mass_width_values, energy):
    """


    Parameters
    ----------
    mass_width_values : ARRAY
        Array containing the mass and width values

    energy : ARRAY


    Returns
    -------
    ARRAY
        Cross Section values converted to nbn

    """

    mass_variable = mass_width_values[0]
    width_variable = mass_width_values[1]
    cross_section = ((12*np.pi/mass_variable**2)*energy**2/\
    ((energy**2-mass_variable**2)**2 + \
     (mass_variable**2 * width_variable**2)))\
        * WIDTH_EE**2

    return cross_section* 0.3894 * 10**6


def breit_wigne_plot(energy_values, cross_section_values, energy_data,\
                     cross_section_data, cross_section_errors):
    """


    Parameters
    ----------
    energy_values : ARRAY
        From filtered (validated) data

    cross_section_values : ARRAY
        Values predicted from Breit Wigne equation

    energy_data : ARRAY
        From filtered (validated) data

    cross_section_data : ARRAY
        From filtered (validated) data

    cross_section_errors : ARRAY
        From filtered (validated) data

    Returns
    -------
    Figure

    Returns plot of obtained and predicted cross section values against energy
    values.

    """


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(energy_values, cross_section_values, linestyle='None',\
                   color='orange', label='Predicted Cross section data')
    ax.scatter(energy_data, cross_section_data, label=\
               'Obtained Cross section data')
    ax.legend()
    ax.errorbar(energy_data, cross_section_data, yerr=cross_section_errors \
                , fmt='o')
    ax.set_xlabel(r'$Energy$ ($GeV$)')
    ax.set_ylabel(r'$Cross Section$ ($nbn$)')
    ax.set_title(r'$Cross Section$ $vs.$ $Energy$ ')
    plt.savefig('Energy Values (GeV) against Cross Section (nbn)', dpi=300)
    plt.show()

    return fig


def chi_squared(mass_width, observed, energy_array, errors):
    """


    Parameter

    ----------
    mass_width : ARRAY

    observed : ARRAY
        Cross-Section experimental data

    energy_array : ARRAY

    errors : ARRAY


    Returns
    -------
    float

    Chi Square

    """

    return np.sum((observed - cross_section_function\
                   (mass_width, energy_array))**2/errors**2)

def reduced_chi_squared(array, mass_width, observed_cross_section, \
                        energy_data, uncertainties):
    """

    Parameters
    ----------
    array : array
        experimental data of energy, cross section and uncertainty on\
            cross section

    mass_width : array
        mass and width of z boson in a 1 x 2 array

    observed_cross_section : array
        Slice of cross section values from array

    energy_data : array
        Slice of energy values from array

    uncertainties : array
        Slice of uncertainties from array

    Returns
    -------
    float

    reduced chi square

    """

    number = len(array)

    return chi_squared(mass_width, observed_cross_section,\
                       energy_data, uncertainties)/(number-2)

def lifetime(width):
    """
    Calculates lifetime in seconds (s) by converting h_bar to GeVs.

    Returns
    -------
    float

    the lifetime of the z boson (in s)

    """
    h_bar_conversion = constants.hbar * 10**(-9) * 1/constants.eV

    return h_bar_conversion/width

def mesh_array(mass, width, array):
    """


    Parameters
    ----------
    mass : FLOAT

    width : FLOAT

    array : ARRAY
        The array of experimental data
        (Energy, cross section, cross section errors)

    Returns
    -------
    mass_array_mesh : N DIMENSIONAL ARRAY

    width_array_mesh : N DIMENSIONAL ARRAY

    chi_square_mesh : N DIMENSIONAL ARRAY

    """

    mass_values = np.linspace(mass-0.055, mass+0.055, 100)
    width_values = np.linspace(width-0.07, width+0.07, 100)

    mass_array_mesh = np.empty((0, len(mass_values)))



    width_array_mesh = np.empty((0, len(width_values)))



    mass_array_mesh, width_array_mesh = np.meshgrid(mass_values, width_values)

    chi_square_mesh = np.zeros((len(mass_values), len(width_values)))

    for i in range(len(mass_array_mesh)):
        for j in range(len(width_array_mesh)):
            chi_square_value = chi_squared([mass_values[i], width_values[j]]\
                                    , array[:, 1], array[:, 0], array[:, 2])
            chi_square_mesh[i][j] = chi_square_value


    return mass_array_mesh, width_array_mesh, chi_square_mesh

def plot_contour(mesh, mass_width_array, minimum_chi_square):
    """


    Parameters
    ----------
    mesh : N DIMENSIONAL ARRAY

    mass_width_array : ARRAY

    minimum_chi_square : FLOAT


    Returns
    -------
    contour_plot : QuadContourSet

    plot of mass and width values against different chi square levels

    """

    parameters_contour_figure = plt.figure()

    parameters_contour_plot = parameters_contour_figure.add_subplot(111)

    parameters_contour_plot.set_title(\
                                r'$\chi^2$ contours against mass and width.',
                                fontsize=14)
    parameters_contour_plot.set_xlabel(r'mass (GeV/$c^2$)', fontsize=14)
    parameters_contour_plot.set_ylabel('width (GeV)', fontsize=14)



    parameters_contour_plot.scatter(mass_width_array[0], mass_width_array[1],\
                                     label='Minimum')

    parameters_contour_plot.contour(mesh[0], mesh[1], mesh[2],\
                    levels=[minimum_chi_square + 1.00], linestyles='dashed',\
                        colors='k')

    chi_squared_levels = (minimum_chi_square + 2.30, minimum_chi_square + 5.99,
                          minimum_chi_square + 9.21)

    contour_plot = parameters_contour_plot.contour(MESH[0], MESH[1], MESH[2], \
                                               levels=chi_squared_levels)
    labels = [r'minimum', r'$\chi^2_{{\mathrm{{min.}}}}+1.00$', \
              r'$\chi^2_{{\mathrm{{min.}}}}+2.30$',\
                  r'$\chi^2_{{\mathrm{{min.}}}}+5.99$',\
                      r'$\chi^2_{{\mathrm{{min.}}}}+9.21$']

    parameters_contour_plot.clabel(contour_plot)

    box = parameters_contour_plot.get_position()
    parameters_contour_plot.set_position([box.x0, box.y0, box.width *\
                                          0.7, box.height])

    for index, label in enumerate(labels):
        parameters_contour_plot.collections[index].set_label(label)
        parameters_contour_plot.legend(loc='center left',\
                                       bbox_to_anchor=(1, 0.5), fontsize=14)

    plt.savefig('contour_plot_of_chi_square.png', dpi=300)

    plt.show()

    return contour_plot

def contour_errors(mesh, reduced_chi_square, mass_width_array):
    """
    CALCULATES UNCERTAINTIES IN MASS,WIDTH AND TAU FROM CONTOUR PLOT
    By FINDING DISTANCE BETWEEN MINIMUM POINT TO MAX MASS/MIN WIDTH OF
    CHI SQUARE + 1 CONTOUR .

    Parameters
    ----------
    mesh : N DIMENSIONAL ARRAY

    reduced_chi_square : FLOAT
        Minimum chi square

    mass_width_array : ARRAY


    Returns
    -------
    uncertainty_mass : FLOAT

    uncertainty_width : FLOAT

    uncertainty_tau : FLOAT

    """

    contour_plot = plot_contour(mesh, mass_width_array, reduced_chi_square)
    path = contour_plot.collections[0].get_paths()[0]
    vertices = path.vertices
    mass = vertices[:, 0]
    width = vertices[:, 1]

    minimum_mass = np.amin(mass)
    minimum_width = np.amin(width)
    maximum_mass = np.amax(mass)
    maximum_width = np.amax(width)

    uncertainty_mass = (maximum_mass - minimum_mass)*0.5
    uncertainty_width = (maximum_width - minimum_width)*0.5

    uncertainty_tau = uncertainty_width/(mass_width_array[1]**2)\
        *lifetime(mass_width_array[1])

    return uncertainty_mass, uncertainty_width, uncertainty_tau

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Main Code~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

array_data_1 = file_validation(read_file(FILE_NUMBER_1))
array_data_2 = file_validation(read_file(FILE_NUMBER_2))

combined_data_with_outliers = array_combined(array_data_1, array_data_2)
validated_data = removing_outliers(combined_data_with_outliers)

ENERGY = validated_data[:, 0]
CROSS_SECTION = validated_data[:, 1]
CROSS_SECTION_ERRORS = validated_data[:, 2]

fit_results = fmin(chi_squared, (90, 3), args=\
            (CROSS_SECTION, ENERGY, CROSS_SECTION_ERRORS), full_output=True)

removing_outliers_2 = validated_data[abs((\
                                    cross_section_function(fit_results[0],\
                            ENERGY)- CROSS_SECTION)/CROSS_SECTION_ERRORS) < 3]

MINIMIZATION_FIT = fmin(chi_squared, (90, 3), args=(removing_outliers_2[:, 1],\
    removing_outliers_2[:, 0], removing_outliers_2[:, 2]), full_output=True)


breit_wigne_plot(removing_outliers_2[:, 0], cross_section_function(\
MINIMIZATION_FIT[0], removing_outliers_2[:, 0]), removing_outliers_2[:, 0],\
                removing_outliers_2[:, 1], removing_outliers_2[:, 2])

MASS_WIDTH_ARRAY = MINIMIZATION_FIT[0]

MESH = mesh_array(MASS_WIDTH_ARRAY[0], MASS_WIDTH_ARRAY[1], \
                  removing_outliers_2)

REDUCED_CHI_SQUARE = chi_squared(MASS_WIDTH_ARRAY, removing_outliers_2[:, 1],\
                        removing_outliers_2[:, 0], removing_outliers_2[:, 2])

ERRORS = contour_errors(MESH, REDUCED_CHI_SQUARE, MASS_WIDTH_ARRAY)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Results~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('The reduced chi squared is', '{0:4.3f}'.format(\
      reduced_chi_squared(removing_outliers_2, MINIMIZATION_FIT[0]\
    , removing_outliers_2[:, 1], removing_outliers_2[:, 0], \
      removing_outliers_2[:, 2])))

print('Mass of Z boson is', '{0:4.2f}'.format(MASS_WIDTH_ARRAY[0]), '+/-',\
      '{0:4.2f}'.format(ERRORS[0]), 'GeV/c^2')

print('Width of Z boson is', '{0:4.3f}'.format(MASS_WIDTH_ARRAY[1]), '+/-',\
      '{0:4.3f}'.format(ERRORS[1]), 'GeV')

print('Lifetime of Z boson is', '{0:3.2e}'.format(lifetime(MASS_WIDTH_ARRAY[1]\
                                )), '+/-', '{0:3.2e}'.format(ERRORS[2]), 's')
