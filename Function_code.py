'''
Created on Jun 8, 2020

@author: colle
'''
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io.fits.column import ColDefs
from astropy.io.fits.hdu.hdulist import HDUList
from xarray.plot.plot import step
#from salsa.UnitConversions import rayleighPerAng2Solstice
from numpy.core.tests.test_umath_accuracy import convert
from calculate_roughness import BRPhase
from scipy.optimize.minpack import curve_fit
import csv
from astropy.io.fits.hdu import hdulist
from matplotlib.pyplot import axis
from astropy.visualization.wcsaxes import ticklabels
from astropy.units import spectral
plt.style.use('classic')
from pyuvis import QUBE
import numpy as np
from pyuvis.io import FUV_PDS
import matplotlib.patches as mpatches
import scipy.interpolate
from astropy.io import fits
from astropy import units as u
import matplotlib.pyplot
import pylab
import os
import glob
from pathlib import Path
from scipy.io import readsav
from astropy import constants as const
from astropy.convolution import convolve, Box1DKernel


def qube_to_color_spectra(data):
    """Code for subsequent data entries."""
    #original color picture of UVIS spectrum
    
    plt.imshow(data.mean(axis=2), aspect='auto', interpolation='nearest',
    cmap=plt.cm.rainbow, extent=[0,120,190.0,111.5])
    
    plt.title('Color Picture of UVIS Spectrum')
    plt.xlabel('Pixel')
    plt.ylabel('Wavelength')
    plt.colorbar()
    
    plt.figure()
    
def background_noise(qube):
    #Calculates the average background noise and the average spectrum
    
    wavelengths = qube.waves.value
    w = np.where((wavelengths >= 1160) & (wavelengths <= 1190))
    
    avg_spectrum = qube.data.mean(axis=2).mean(axis=1)
    avg_background = np.mean(avg_spectrum[w])
     
    return avg_background

def single_px_noise_subbed(qube, x_pixel, y_pixel):
    #function with all variables that will go into plot

    spectrum = qube.data[:512, x_pixel , y_pixel ]
    avg_noise = background_noise(qube)
    spectrum_noise_subtracted = spectrum - avg_noise
    
    return spectrum, avg_noise, spectrum_noise_subtracted

def plot_single_pixel(spectrum_noise_subtracted, spectrum, waves):
    #function plotting the Uncalibrated single pixel spectrum
     
    plt.plot(waves, spectrum)
    
    plt.title('Uncalibrated UVIS Spectrum')
    plt.xlabel('Wavelength in $\AA$')
    plt.ylabel('Raw Counts')
    
    plt.plot(waves, spectrum_noise_subtracted)
    
    blue_patch = mpatches.Patch(color='blue', label='Original Data')
    green_patch = mpatches.Patch(color='green', label='Noise Subtracted')
    plt.legend(handles=[green_patch, blue_patch], loc='upper right') 
    
    plt.figure()
    
def cal_matrix_noise_subbed(qube):
    #This is the calibration matrix with the noise being subtracted and the evil pixels located that will be called in the next function.
    
    cal_matrix = qube.cal_matrix.data[:512, :, :].reshape(512, 64)
    avg_noise = background_noise(qube)
    data = qube.data[:512, :, :]
    qube_noise_subbed = data - avg_noise
    evil_px = np.where(cal_matrix == -1)
    
    return cal_matrix, qube_noise_subbed, evil_px


    
def plot_cal_matrix(cal_matrix, qube_noise_subbed, evil_px):
    #this does a separate plot of the calibration matrix with the noise subtracted.
    
    arrcal_fuv = get_calibrated_cube(cal_matrix, qube_noise_subbed, evil_px)
    qube_to_color_spectra(arrcal_fuv)
    
    return arrcal_fuv

def get_calibrated_cube(cal_matrix, qube_noise_subbed, evil_px):
    #this does a separate plot of the calibration matrix with the noise subtracted.
    
    cal_matrix[evil_px] = np.nan
    
    arrcal_fuv = np.copy(qube_noise_subbed)
    for i in range(0, arrcal_fuv.shape[2]):
        arrcal_fuv[:, :, i] = arrcal_fuv[:, :, i] * cal_matrix
    
    
    return arrcal_fuv

def nan_helper(arrcal_fuv):
    #Function that removes NaN's (to be put in another function for execution)
    return np.isnan(arrcal_fuv), lambda z: z.nonzero()[0]
   
def interp_data(arrcal_fuv, nan_helper):
    #removes NaN's and interpolates the data by looping through all the data in the cube.
     
    for i in range(0, arrcal_fuv.shape[2]):
        for j in range(0, arrcal_fuv.shape[1]):
            temp_array = arrcal_fuv[:, j, i]
            nans, x = nan_helper(temp_array)
            temp_array[nans]= np.interp(x(nans), x(~nans), temp_array[~nans])
            arrcal_fuv[:, j, i] = temp_array
   
    return arrcal_fuv
    
def interp_img(arrcal_fuv, data):
    #images of the original and interpolation calibration on the 0 axis.
    
    plt.imshow(arrcal_fuv.mean(axis=0)[2:61, :], aspect='auto', interpolation='nearest',
    cmap=plt.cm.rainbow)
    
    plt.title('Calibrated Image')
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')
    clb = plt.colorbar()
    clb.set_label('Kilorayleighs/$\AA$')
    
    plt.figure()
    
    
    plt.imshow(data.mean(axis=0)[2:61, :], aspect='auto', interpolation='nearest',
    cmap=plt.cm.rainbow)
    
    plt.title('Uncalibrated Image')
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')
    clb = plt.colorbar()
    clb.set_label('Raw Counts')
    
    plt.figure()
    
def interp_plot(b_ring_spectrum, waves):    #########################################################################################
    #Calibrated UVIS Spectrum plot
    
    plt.plot(waves, b_ring_spectrum)
    
    plt.title('Calibrated UVIS Spectrum')
    plt.xlabel('Wavelengths')
    plt.ylabel('Kilorayleighs/$\AA$')
    
    green_patch = mpatches.Patch(color='blue', label='New Data')
    plt.legend(handles=[green_patch], loc='upper right')
   
    
    plt.figure()
    
def fits_data(hdul, col_name):
    #this takes a FITS file, locates which directory you are looking for, takes the data and goes into the next function.
    #col_name is replaced in the main program as the specific thing you are looking at in this FITS file.
    
    #hdul.info() #gives you all the info of the FITS file in your console
    table = hdul[3]
    image = table.data.field(col_name).transpose()
    
    return image
       
def fits_reader_imgshow(hdul, col_name):
    #this function creates an image of the FITS file data depending on what col_name is brought up.
    #col_name is replaced in the main program by the specific column you are wanting to look at in the FITS file.
    fits_image = fits_data(hdul, col_name)
    
    #plots the image arrays of the FITS file
    plt.imshow(np.squeeze(fits_image), aspect='auto')
    
    plt.title('Center Ring Plane Radii')
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')
    
    plt.colorbar()
    
def get_b_ring_pixel(fits_image):
    #read image and return b ring array info. 
    
    return np.where((fits_image >= 92000) & (fits_image <= 117000))
    

def get_b_ring_spectra(cal_cube, fits_image):
    #loop over all b ring pixel locations
    #get the spectrum from each of those locations
    #store those all in a python list
    #return the list
    
    b_ring_px = get_b_ring_pixel(fits_image)
    x = b_ring_px[0]
    y = b_ring_px[1]
    spectra_list = []
    
    for i in range(0, b_ring_px[0].size):
        cal_spectrum = cal_cube[:512, x[i], y[i]]
        spectra_list.append(cal_spectrum)
        
    return spectra_list



def get_b_ring_mu0(b_ring_px, hdu):
    
    x = b_ring_px[0]
    y = b_ring_px[1]
    
    # Get the pixel center incidence angle, take the cos to get mu0 (see Hapke)
    incidence_angles = fits_data(hdu, 'PIXEL_CENTER_INCIDENCE_ANGLE')
    
    mu0 = []
    for i in range(0, b_ring_px[0].size):
        # Get the pixel center incidence angle, take the cos to get mu0 (see Hapke)
        incidence_angle = incidence_angles[x[i], y[i]]
        if incidence_angle > 90:
            # The 180 - is needed because the incidence angle is measured from Saturn's North direction.
            incidence_angle = (180.0 - incidence_angle) * np.pi / 180.0 # Convert to radians
        mu0.append(np.cos(incidence_angle))
        
    return np.array(mu0)
        
def plot_b_ring_spectra(cal_cube, fits_image, wavelengths):
    #loop over all b ring spectra that your passing in
    #plot the spectra
    #all plot into the same plot window
    #cal_cube, fits_image, & wavelengths all get something passed into them in the main program.
    
    spectra_list = get_b_ring_spectra(cal_cube, fits_image)
    plt.figure()
    
    for i in range(len(spectra_list)):
        plt.plot(wavelengths, spectra_list[i])
        plt.title('Plot of B Ring Spectra')
        plt.xlabel('Wavelength($\AA$)')
        plt.ylabel('Brightness')
    
def get_phase_angle_pixel(fits_image):
    #read image and return b ring phase angle array info. 
    #92000-117000 measured in km
    
    return np.where((fits_image >= 92000) & (fits_image <= 117000))
    
def get_phase_angle_array(b_ring_px, fits_image):
    #loop over all b ring pixel locations for phase angle
    #get the spectrum from each of those locations
    #store those all in a python list
    #return the list
    
    x = b_ring_px[0]
    y = b_ring_px[1]
    phase_angle_list = []
    
    for i in range(0, b_ring_px[0].size):
        phase_angle = fits_image[x[i], y[i]]
        phase_angle_list.append(phase_angle[0])
        
    return np.array(phase_angle_list)

    
def plot_phase_angle_values(b_ring_px, fits_image):
    #Plots a histogram of the Phase angle values
    phase_angle_array = get_phase_angle_array(b_ring_px, fits_image)
    num_bins = 10

    fig, ax = plt.subplots()
    ax.hist(phase_angle_array, num_bins, density=1, color='pink')
    ax.set_title('Histogram of Phase Angle Values')
    ax.set_xlabel('Phase Angle')
    ax.set_ylabel('Number of Observations')
   
    
def get_range_in_wavelength(L_wavelength, U_wavelength, wavelengths):
    #Location in wavelength array that are inside specified upper and lower wavelength
    #U-wavelength & L_wavelength are in Angstroms.
    
    w = np.where((wavelengths.value <= U_wavelength) & (wavelengths.value >= L_wavelength))
    return w

def get_brightness(spectra_list, L_wavelength, U_wavelength, wavelengths):
    #takes the average brightness values and puts it into a python list.
    #must specify what wavelengths you want in the main program when it is called. 
    
    w = get_range_in_wavelength(L_wavelength, U_wavelength, wavelengths)
    avg_brightness_list = []
    
    for i in range(0, len(spectra_list)):
        avg_brightness = np.mean(spectra_list[i][w])
        avg_brightness_list.append(avg_brightness)
    
    return np.array(avg_brightness_list)

def remove_zeros(avg_brightness_list):
    #remove all zeroes from scatter plot
    w = np.where(avg_brightness_list > 0)
    
    return w

def convert_spectrum_to_reflectance(wavelengths, uvis_spectrum, solar_irradiance, mu0):
    '''
    Function to convert UVIS spectrum to solar_irradiance.
    Params: 
        wavelengths : Wavelengths in Angstroms
        uvis_spectrum : UVIS spectrum in KR/A
        solar_irradiance : SOLSTICE spectrum in W/m^2/nm
    '''
    
    # Convert to SI units
    wavelengths_in_meters = wavelengths.value / 10.0**10 # Convert from Angstroms to meters.
    const.h * const.c / wavelengths_in_meters
    
    # Calculate the energy of each photon at each wavelength
    # using the standard formula E = h*c/lambda, h = Planck's constant, c = speed of light, lambda = wavelength
    photon_energy = const.h * const.c / wavelengths_in_meters
    
    # Using the UVIS user's guide definition of KR.
    # 1 KR / A = 10^9 photons / cm^2 / s / A / str
    #          = 10^14 photons / m^2 / s / nm / str
    uvis_spectrum *= 10.0**14 * photon_energy.value / (4 * np.pi)
    
    # Take into account ring-lighting geometry (See Bradley's chapter in UVIS Users Guide)
    solar_irradiance = solar_irradiance * mu0
    
    # Divide the UVIS spectrum by the SOLSTICE spectrum
    return uvis_spectrum / solar_irradiance

def get_phase_curve(phase_angle_image, mu0, b_ring_px, spectra_list, L_wavelength, U_wavelength, wavelengths, solar_irradiance):
    x = get_phase_angle_array(b_ring_px, phase_angle_image)
    
    # Convert the units of each spectrum to reflectance: I/F
    i_over_f_spectra = [] 
    for i in range(len(spectra_list)):
        i_over_f = convert_spectrum_to_reflectance(wavelengths, spectra_list[i], solar_irradiance, mu0[i])
        
        i_over_f_spectra.append(i_over_f)
    
    y = get_brightness(i_over_f_spectra, L_wavelength, U_wavelength, wavelengths)
    
    return x, y
    
def get_solar_spectrum(file):
    #Gets the solar spectrum from SORCE SOLSTICE data
    data = readsav(file)
    wavelength = data.sorce.wavelength[0]
    irradiance = data.sorce.irradiance[0]
    return wavelength, irradiance

def plot_solar_spectrum(file):
    #Function that plots the Solar spectrum
    wavelength, irradiance = get_solar_spectrum(file)
    plt.plot(wavelength, irradiance, color='#8A2BE2')
    plt.title('SORCE SOLSTICE FUV Spectrum', fontweight='bold', fontsize='18', color='#008080')
    plt.xlabel('Wavelength [nm]', fontsize='14', color='#0000FF')
    plt.ylabel('Irradiance [W/m^2/nm]', fontsize = '14', color='#0000FF')
    

def plot_bright_vs_phase_angle(x, y, color, marker, label): #########################################################################
    #plot both brightness and phase angle together on same plot. 
    #x- phase angle
    #y- brightness as 1 dimension
    #will show a piece of a phase curve
    
    #phase_angles, brightness_values, R  = best_fit_phase_curve(path, files, wavelength, color, marker)
    
    w = remove_zeros(y)
    x = x[w]
    y = y[w]
    
        
    plt.scatter(x, y, alpha=0.6, color=color, marker=marker, label=label)
    
    plt.title('Scatter Plot of Reflectance and Phase Angle', fontweight='bold', fontsize='18', color='#008080')
    plt.xlabel('Phase Angle (in degrees)', fontsize='14', color='#0000CD')
    plt.ylabel('Reflectance (I/F)', fontsize = '14', color='#0000CD')
    plt.legend(loc='upper right')
    
    
    
def plot_slope_func_of_WL(x, y, wavelength_list, files):
        
    slopes = np.zeros_like(wavelength_list)
    intercepts = np.zeros_like(wavelength_list)
    index = 0
    
    for i in wavelength_list:   
        qube = FUV_PDS() 
        hdul, radius_img, b_ring_px, b_ring_spectra_calc, phase_angle_img, mu0 = return_b_ring_info(files, qube) 
        arrcal_fuv_interp = return_calib_cube(qube)
        spectra_list = get_b_ring_spectra(arrcal_fuv_interp, phase_angle_img)
        x = get_phase_angle_array(b_ring_px, phase_angle_img)
        y = get_brightness(spectra_list, i, i+10, qube.waves)
        w = remove_zeros(y)
        x = x[w]
        y = y[w]
        m, b = np.polyfit(x, y, 1) #m = slope, b = intercept.
        slopes[index] = m
        intercepts[index] = b
        index = index + 1
        
        
    plt.plot(wavelength_list, slopes) 
    plt.title('Slopes as a Function of Wavelength')
    plt.xlabel('Wavelength')
    plt.ylabel('Slope')
    plt.legend(loc='best')
        
def loop_over_files(path, files, L_wavelength, U_wavelength, color, marker):
    #loops over all the .DAT files needing to be analyzed and also gets the corresponding .fits files.
    #color and marker must be specified in main program
    phase_angles = np.zeros(shape = (0,))
    brightness_values = np.zeros(shape = (0,))
    for dat_file in files:
        basename = os.path.basename(dat_file)[:-4]
        fitsfile = path + "UVISGeometry" + os.sep + basename + ".fits"
        
        # Retrieve the solar spectrum file name.
        solar_file = path + "SOLSTICESpectra" + os.sep + "solstice_hr_" + basename[3:-6] + ".sav"
        
        # If the fits file or solar spectrum file are not present, jump to the next file.
        if not os.path.exists(fitsfile) or not os.path.exists(solar_file):
            continue
        
        qube = FUV_PDS(dat_file)
        hdul, radius_img, b_ring_px, b_ring_spectra_calc, phase_angle_img, mu0 = return_b_ring_info(fitsfile, qube)
        
        # Get Solar Spectrum
        solar_wavelength, solar_irradiance = get_solar_spectrum(solar_file)
        
        # Get the phase curve
        x, y = get_phase_curve(phase_angle_img, mu0, b_ring_px, b_ring_spectra_calc, L_wavelength, U_wavelength, qube.waves, solar_irradiance)
        
        phase_angles = np.append(phase_angles, x)
        brightness_values = np.append(brightness_values, y)
        
    #plot_bright_vs_phase_angle(phase_angles, brightness_values, color, marker, str(L_wavelength) + ' $\AA$')
    
    return phase_angles, brightness_values   
        
def return_calib_cube(qube):
    #new function that returns the calibrated cube.
    
    cal_matrix, qube_noise_subbed, evil_px = cal_matrix_noise_subbed(qube)
    arrcal_fuv = get_calibrated_cube(cal_matrix, qube_noise_subbed, evil_px)
    arrcal_fuv_interp = interp_data(arrcal_fuv, nan_helper)

    
    return arrcal_fuv_interp

def return_b_ring_info(file, qube):
    hdul = fits.open(file)
    radius_img = fits_data(hdul, 'CENTER_RING_PLANE_RADII')
    b_ring_px = get_b_ring_pixel(radius_img)
    arrcal_fuv_interp = return_calib_cube(qube)
    b_ring_spectra_calc = get_b_ring_spectra(arrcal_fuv_interp, radius_img)
    phase_angle_img = fits_data(hdul, 'PIXEL_CENTER_PHASE_ANGLE')
    
    # Get the pixel center incidence angle, take the cos to get mu0 (see Hapke)
    mu0 = get_b_ring_mu0(b_ring_px, hdul)
    
    return hdul, radius_img, b_ring_px, b_ring_spectra_calc, phase_angle_img, mu0

def best_fit_phase_curve(path, files, wavelength, color, marker):
    #plots the best fit phase curve
    phase_angles, brightness_values = loop_over_files(path, files, wavelength, wavelength+10, color, marker)
    bounds = ([0.1],[80.0])
    p0 = (30.0,)
    popt, pcov = curve_fit(lambda phase, rough: BRPhase(phase, rough, wavelength=wavelength*1e-4), phase_angles[::10], brightness_values[::10], p0=p0, bounds=bounds)
    plot_bright_vs_phase_angle(phase_angles, brightness_values, '#8A2BE2', 'x', str(wavelength) + ' $\AA$')
    phase_angles = np.linspace(1, 179, 179)
    R = BRPhase(phase_angles, popt[0], wavelength*1e-4)
    plt.plot(phase_angles, R, color='blue', label='B-Ring Best-Fit : $\\bar{\\Theta}=' + str(int(popt[0])) + '\\degree$')
    plt.ylim((0, 0.5))
    plt.legend(loc='upper right')
    return phase_angles, brightness_values, R  
       

def plot_all_phase_curves(phase_curve_data_files):
    #plots all the phase curves into one plot window.
    
    for file in phase_curve_data_files:
        phase_angles = []
        R = []
        with open(file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row) == 3:
                    phase_angles.append(float(row[0]))
                    R.append(float(row[2]))
          
        
        plt.plot(phase_angles, R, label=file.split("_")[-1].split(".")[0] + ' $\\AA$')
            
        plt.xlabel('Phase Angle (in degrees)', fontsize = '14', color='#0000FF')
        plt.ylabel('Reflectance [I/F]', fontsize = '14', color='#0000FF')
        plt.title('Reflectance vs. Phase Angle', fontweight='bold', fontsize='18', color='#9400D3')
            
      
    plt.legend(loc='upper right')
    plt.savefig('all_phase_curves.png')
    plt.ylim((0, .025))
    plt.savefig('all_phase_curves_zoomed.png')
    
    
def get_avg_b_ring_spectrum(mu0, spectra_list, wavelengths, solar_irradiance):
      
    i_over_f = np.zeros_like(wavelengths.value)
    for i in range(len(spectra_list)):
        temp = convert_spectrum_to_reflectance(wavelengths, spectra_list[i], solar_irradiance, mu0[i])
        i_over_f += temp
    i_over_f /= len(spectra_list)
    
    return i_over_f
        
def plot_avg_reflectance_spectrum(wavelengths, i_over_f): ##########################################################################################****
    
    plt.plot(wavelengths, i_over_f, color='#8A2BE2')
    plt.title('B-Ring Reflectance', fontweight='bold', fontsize='18', color='#008080')
    plt.xlabel('Wavelength in $\\AA$', fontsize='14', color='#0000FF')
    plt.ylabel('Reflectance [I/F]', fontsize = '14', color='#0000FF')
   
   
    plt.ylim((0, 0.1))
    plt.xlim((1200, 1900))
    
def plot_avg_radiance_spectrum(spectra_list, wavelengths):
    
    radiance = np.zeros_like(wavelengths.value)
    for i in range(len(spectra_list)):
        radiance += spectra_list[i]
    radiance /= len(spectra_list)
    
    interp_plot(radiance, wavelengths)
    return radiance   
    

    
    
    
if __name__ == '__main__':
    
    
    
    
    dat_file = r'C:\Users\colle\UVISData\FUV2005_230_15_15.DAT'
    fits_file = r'C:\Users\colle\UVISGeometry\FUV2005_230_15_15.fits'
    qube = FUV_PDS(dat_file)
    path = "C:\\users\\colle\\"
    basename = os.path.basename(dat_file)[:-4]
    solar_file = path + "SOLSTICESpectra" + os.sep + "solstice_hr_" + basename[3:-6] + ".sav"
    arrcal_fuv_interp = return_calib_cube(qube)
  
    
    
    hdul, radius_img, b_ring_px, b_ring_spectra_calc, phase_angle_img, mu0 = return_b_ring_info(fits_file, qube)
    spectra_list = get_b_ring_spectra(arrcal_fuv_interp, radius_img)
    wavelengths, solar_irradiance = get_solar_spectrum(solar_file)
    i_over_f = get_avg_b_ring_spectrum(mu0, spectra_list, qube.waves, solar_irradiance)
    plot_avg_reflectance_spectrum(qube.waves, i_over_f)
    
    plt.show()
    
    
    #plt.figure()
    #plot_solar_spectrum(solar_file)
    #plt.figure()
    
    #plot_avg_radiance_spectrum(spectra_list, qube.waves)
   
    #plot_bright_vs_phase_angle(x, y, color, marker, label)
    #wavelengths = np.array([1200, 1210, 1600, 1700, 1800, 1900], dtype=np.double)
    #phase_curve_data_files = glob.glob(path + "eclipse-workspace"+ os.sep + "phase_curve_of_saturns_rings"+ os.sep + "phase_curve_*.0.csv")
    #phase_curve_data_files = [file for file in phase_curve_data_files if '.csv' in file]
    #plot_all_phase_curves(phase_curve_data_files)
    
    #plt.show()

#     plt.show()   
    #DAT files start getting read here
   
    #files = glob.glob(path + "UVISdata" + os.sep + "FUV*.DAT")
    #files = [f for f in files if '.DAT' in f and 'CAL_3' not in f]

    
    #loop_over_files(path, files, 1800, 1810, '#B22222', 'x')
    
    #wavelengths = np.array([1200, 1210, 1600, 1700, 1800, 1900], dtype=np.double)
    #for wavelength in wavelengths:
        #plt.figure()
        #print("Processing wavelength " + str(wavelength))
        #phase_angles, brightness_values, R = best_fit_phase_curve(path, files, wavelength, '#8A2BE2', 'x')
        #with open('phase_curve_' + str(wavelength) + '.csv', 'w') as csvfile:
            #writer = csv.writer(csvfile)
            #for (p, b, r) in zip(phase_angles, brightness_values, R):
                #writer.writerow([p, b, r])
        
        #plt.savefig('phase_curve_' + str(wavelength) + '.png')
    #plt.show()
   
    
    