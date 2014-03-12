# --------------------------    LIBRARY and MODULE IMPORTS  --------------------

import sys
import ephem
from pylab import *
from os import system
from math import log
from scipy import average, std, median, random, zeros, clip, array, append, mean
from scipy import histogram, optimize, sqrt, pi, exp, where, split
from scipy import sum, std, resize, size, isnan
from scipy import optimize, shape, indices, arange, ravel, sqrt
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy import loadtxt, savetxt, sort
import pyfits
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import operator

# --------------------------    USER INPUT PARAMETERS   --------------------

progenitor_image_name = "j_long_SN.266.3_coadd.fits"
region_file = "SN.266.3.reg"

sextractor_bin = "/usr/local/bin/sex"
# weight_image_name = progenitor_image_name.replace("coadd", "coadd.weight")
weight_image_name = "j_long_SN.266.3_coadd.weight.fits"

# --------------------------    FUNCTION DEFINITIONS    ------------------------
# Calculate the residual for the gaussian fit.
def gaussian_residuals(params, y, x):
    central_height, center_x, x_width = params
    err = y - ((central_height/sqrt(2*pi*x_width**2))*
        2.71828182845**((-1*(x-center_x)**2)/(2*x_width**2)))
    return err
# Fit the input data with a gaussian curve.
def fit_gaussian(y, x):
    input_params = [5000.0, 0.0, 30.0]
    fit_params, success = optimize.leastsq(gaussian_residuals, input_params, 
        args=(y, x), maxfev=10000)
    return fit_params
# Call Source Extractor for photometry on science image.
def make_sex_cat(progenitor_image_name, weight_image_name, aperture_size):
    sexcat_file = progenitor_image_name.replace(".fits", ".sex")
    min_pix = pi*(aperture_size*aperture_size/4)/4
    system(sextractor_bin + " " + progenitor_image_name + " -c pairitel_photo.sex " + 
        "-DETECT_MINAREA " + str(min_pix) +
        " -BACK_SIZE 64 " + 
        "-BACK_FILTERSIZE 3 " + 
        "-BACK_TYPE AUTO " + 
        "-BACK_VALUE 0.0 " +
        "-FILTER N " +
        "-PHOT_APERTURES " + str(aperture_size) + 
        " -WEIGHT_IMAGE " + weight_image_name + 
        " -CATALOG_NAME " + sexcat_file)
    return sexcat_file
# Calculate the heliocentric julian date.
def heliocentric_julian_date(observation_time, observation_ra_radians, 
    observation_dec_radians):
    # Compute the observation time in Heliocentric Julian Date. First convert to 
    # julian date (jd) by adding the offset to ephem.date format.
    observation_time_jd = float(observation_time) + 2415020.0
    # Calculate the Sun's position in the sky.
    sun = ephem.Sun()
    sun.compute(observation_time)
    sun_ra_radians = float(sun.a_ra)
    sun_dec_radians = float(sun.a_dec)
    # Calculate the Earth-Sun light travel time in days.
    earth_sun_light_travel_time = sun.earth_distance * 0.00577551833
    # Calculate the observation time in heliocentric julian date.
    observation_time_hjd = (observation_time_jd - earth_sun_light_travel_time * 
        (sin(observation_dec_radians) * sin(sun_ra_radians) + 
        cos(observation_dec_radians) * cos(sun_ra_radians) * 
        cos(observation_dec_radians - sun_dec_radians)))
    return observation_time_hjd
# Median Absolute Deviation clipping for input list of numbers.
def mad_clipping(input_data, sigma_clip_level):
    medval = median(input_data)
    sigma = 1.48 * median(abs(medval - input_data))
    high_sigma_clip_limit = medval + sigma_clip_level * sigma
    low_sigma_clip_limit = medval - sigma_clip_level * sigma
    clipped_data = []
    for value in input_data:
        if (value > low_sigma_clip_limit) and (value < high_sigma_clip_limit):
            clipped_data.append(value)
    clipped_data_array = array(clipped_data)
    new_medval = median(clipped_data_array)
    new_sigma = 1.48 * median(abs(medval - clipped_data_array))
    return clipped_data_array, new_medval, new_sigma

def moffat_residuals(params, z, x, y):
    central_height, center_x, center_y, alpha, beta = params
    r2 = (x - center_x)**2 + (y - center_y)**2
    err = z - (central_height * (1 + (r2/(alpha**2)))**(-beta))
    return err
    

def fit_moffat(z, x, y):
    """Returns (central_height, center_x, center_y, alpha, beta
    the moffat parameters of a 2D distribution found by a fit"""
    input_params = [15000.0, 5.0, 5.0, 5.0, 5.0]
    fit_params, success = optimize.leastsq(moffat_residuals, input_params, 
        args=(z, x, y), maxfev=10000)
    return fit_params
    
def fit_fwhm(sat_locations, objects_data, fwhm, fwhm_stdev):
    fwhm_list = []
    for location in sat_locations:
        m = location[0] # vertical from bottom
        n = location[1] # horizontal from left
        submask_data = objects_data[m-5:m+5,n-5:n+5]
        z = []
        x = []
        y = []
        x_y = []
        r = []
        for i in range(len(submask_data)):
            i_pixel = i + 0
            for j in range(len(submask_data[i])):
                if submask_data[i][j] != 0:
                    j_pixel = j + 0
                    x.append(j_pixel)
                    y.append(i_pixel)
                    x_y.append([j_pixel, i_pixel])
                    z.append(submask_data[i][j])
        try:
            fit_params = fit_moffat(z, x, y)
        except(TypeError):
            continue
        central_height, center_x, center_y, alpha, beta = fit_params
        for coordinate in x_y:
            r.append(sqrt((coordinate[0]-center_x)**2 + 
                (coordinate[1]-center_y)**2))
        fit_data = []
        residual_data = []
        abs_residual_data = []
        for k in range(len(r)):
            fit_point = (central_height * (1 + (r[k]/alpha)**2)**(-beta))
            fit_data.append(fit_point)
            residual_data.append(z[k] - fit_point)
            abs_residual_data.append(abs(z[k] - fit_point))
    
        metric = mean(abs_residual_data) / central_height # Lower is better
        fwhm_unqiue = 2*alpha*sqrt(2**(1/beta) - 1)
        if ((metric < 0.02) and (fwhm_unqiue > (fwhm - fwhm_stdev)) and 
            (fwhm_unqiue < (fwhm + fwhm_stdev))):
#             print ("Finished fit for saturated star at " + str(m) + " " 
#                 + str(n) + ". FWHM: " + str(fwhm_unqiue) + " with metric: " + str(metric))
            fwhm_list.append(fwhm_unqiue)
    fwhm = mean(fwhm_list)
    fwhm_stdev = std(fwhm_list)
    return fwhm, fwhm_stdev
    

# --------------------------    BEGIN PROGRAM   --------------------------------


# Run source extractor to find the pixel locations of stars.
print "Running Source Extractor to find locations of stars."
system(sextractor_bin + " -c pairitel_photo.sex " + progenitor_image_name + 
    " -WEIGHT_IMAGE " + progenitor_image_name.replace(".fits", ".weight.fits") + 
    " -PARAMETERS_NAME star_find.param " + 
    "-CATALOG_NAME star_cat.txt " + 
    "-DETECT_MINAREA 8 " +
    "-BACK_SIZE 64 " + 
    "-BACK_FILTERSIZE 3 " + 
    "-BACK_TYPE AUTO " + 
    "-BACK_VALUE 0.0 " +
    "-FILTER N " +
    "-CHECKIMAGE_TYPE OBJECTS " + 
    "-CHECKIMAGE_NAME check.fits " +
    "-PHOT_APERTURES 5.8")


objects_image = "check.fits"
hdulist = pyfits.open(objects_image)
objects_data = hdulist[0].data
hdulist.close()
# system("rm check.fits")

sex_file = file("star_cat.txt", "r")
sat_locations = []
for line in sex_file:
    if (line[0] == "#"):
        continue
    else:
        if float(line.split()[2]) == 0:
            x_index = int(round(float(line.split()[0]) - 1))
            y_index = int(round(float(line.split()[1]) - 1))
            sat_locations.append([y_index, x_index])


fwhm, fwhm_stdev = fit_fwhm(sat_locations, objects_data, 2.8, 0.8)
print "Initial fwhm fit complete."
print ("FWHM mean: " + str(fwhm) + " stdev: " + str(fwhm_stdev) + ".")
fwhm, fwhm_stdev = fit_fwhm(sat_locations, objects_data, fwhm, fwhm_stdev)
print "Sigma clipping of fwhm complete."
print ("FWHM mean: " + str(fwhm) + " stdev: " + str(fwhm_stdev) + ".")

aperture_size = (1.5)*fwhm
print "Aperture size: ", aperture_size
if isnan(aperture_size):
    print "Aperture size not found, using default 4.5."
    aperture_size = 4.5



# Now we do photometry. We use the region file to extract the target ra and dec.
region_file = file(region_file, "r")
for line in region_file:
    if line[:6] == "circle":
        target_ra_deg = float(line.split("(")[1].split(",")[0])
        target_dec_deg = float(line.split("(")[1].split(",")[1])
target_ra_rad = target_ra_deg * 0.01745329252
target_dec_rad = target_dec_deg * 0.01745329252

band = (progenitor_image_name.split("_")[0])

# We collect the 2MASS photometry of field sources with a query to vizier.
# Most of this is forming the query and then formatting the returned text file.
viz_input_file = file("viz_input.txt", "w")
if band == "j":
    viz_input_file.write(
        "-source=2MASS/PSC\n" + 
        "-c=" + str(target_ra_deg) + " " + str(target_dec_deg) + "\n" +
        "-c.rm=15\n" +
        "-out=_r RAJ2000 DEJ2000 " + 
            "Jmag e_Jmag Jsnr " + 
            "Hmag e_Hmag Hsnr " + 
            "Kmag e_Kmag Ksnr " + 
            "Qflg\n" +
        "-sort=_r\n" + 
#        "Qflg==AAA\n" + 
        "Jsnr=>20"
        + "\nJmag=>9.7"
        )
elif band == "h":
    viz_input_file.write(
        "-source=2MASS/PSC\n" + 
        "-c=" + str(target_ra_deg) + " " + str(target_dec_deg) + "\n" +
        "-c.rm=15\n" +
        "-out=_r RAJ2000 DEJ2000 " + 
            "Jmag e_Jmag Jsnr " + 
            "Hmag e_Hmag Hsnr " + 
            "Kmag e_Kmag Ksnr " + 
            "Qflg\n" +
        "-sort=_r\n" + 
#        "Qflg==AAA\n" + 
        "Hsnr=>20"
        + "\nHmag=>9.7"
        )
elif band == "k":
    viz_input_file.write(
        "-source=2MASS/PSC\n" + 
        "-c=" + str(target_ra_deg) + " " + str(target_dec_deg) + "\n" +
        "-c.rm=15\n" +
        "-out=_r RAJ2000 DEJ2000 " + 
            "Jmag e_Jmag Jsnr " + 
            "Hmag e_Hmag Hsnr " + 
            "Kmag e_Kmag Ksnr " + 
            "Qflg\n" +
        "-sort=_r\n" + 
#        "Qflg==AAA\n" + 
        "Ksnr=>20"
        + "\nKmag=>9.7"
        )
viz_input_file.close()
system("vizquery -mime=csv -site=cfa viz_input.txt > viz_output.txt")
system("rm -f viz_input.txt")
viz_output_cropped_file = file("viz_output_cropped.txt", "w")
viz_output_file = file("viz_output.txt", "r")
line_num = 0
for line in viz_output_file:
    line_num += 1
    if line_num > 46 and line != "\n" and line[0] != "#":
        if band == "j":
            try:
                if line.split(";")[12][0] == "A":
                    viz_output_cropped_file.write(line)
            except:
                continue
        elif band == "h":
            try:
                if line.split(";")[12][1] == "A":
                    viz_output_cropped_file.write(line)
            except:
                continue
        elif band == "k":
            try:
                if line.split(";")[12][2] == "A":
                    viz_output_cropped_file.write(line)
            except:
                continue
viz_output_cropped_file.close()
viz_output_file.close()
# Define the obj_string and band_type from the progenitor_image_name. If the 
# progenitor_image_name is not in the format used by PARITIEL reduction
# pipeline 3, these will not work properly.
obj_string = progenitor_image_name.split("_")[2]
band_type = (progenitor_image_name.split("_")[0] + "_" + 
    progenitor_image_name.split("_")[1])
# Run source extractor on the science image.
sexcat = make_sex_cat(progenitor_image_name, weight_image_name, aperture_size)
# Store placehold values for the target photometry.
target_mag = 999
target_e_mag = 999
# Create the catalog of 2MASS stars from the vizier data which corrresponds to
# the filter/band used for the science image.
vizcat_file = file("viz_output_cropped.txt", "r")
vizcat_starlist = []
for line in vizcat_file:
    data_list = line.rstrip().lstrip().split(";")
    ra = float(data_list[1]) # degrees
    dec = float(data_list[2]) # degrees
    if band == "j":
        mag = float(data_list[3])
        e_mag = float(data_list[4])
        snr = float(data_list[5])
    if band == "h":
        mag = float(data_list[6])
        e_mag = float(data_list[7])
        snr = float(data_list[8])
    if band == "k":
        mag = float(data_list[9])
        e_mag = float(data_list[10])
        snr = float(data_list[11])
    vizcat_starlist.append([ra, dec, mag, e_mag, snr])
vizcat_file.close()
# Create the sexcat_starlist from the Source Extractor output catalog. Also fill
# in the sex_inst_mag_list which will be used as a diagnostic check on the 
# computed upper limit.
sexcat_starlist = []
sex_inst_mag_list = []
sexcat_file = file(sexcat, "r")
for line in sexcat_file:
    ra = float(line.split()[0])
    dec = float(line.split()[1])
    mag = float(line.split()[2])
    mag_err = float(line.split()[3])
    flags = int(line.split()[4])
    sexcat_starlist.append([ra, dec, mag, mag_err, flags])
    if flags == 0:
        sex_inst_mag_list.append([mag, mag_err, ra, dec])
sexcat_file.close()
# Compare the entries in sexcat_starlist and vizcat_starlist to create a 
# combined_starlist which has entries for sources with both archival and new
# instrumental data. The target need not be included in the archive.
combined_starlist = []
for sexcat_star in sexcat_starlist:
    sexcat_star_ra_rad = sexcat_star[0] * 0.01745329252
    sexcat_star_dec_rad = sexcat_star[1] * 0.01745329252
    target_separation_arcsec = 206264.806247*(float(ephem.separation(
        (target_ra_rad, target_dec_rad), 
        (sexcat_star_ra_rad, sexcat_star_dec_rad))))
    if target_separation_arcsec < 5:
        combined_starlist.append([sexcat_star[0], sexcat_star[1], 
            999, 999, 
            sexcat_star[2], sexcat_star[3], 
            sexcat_star[4]])
        continue
    for calibcat_star in vizcat_starlist:
        calibcat_star_ra_rad = calibcat_star[0] * 0.01745329252
        calibcat_star_dec_rad = calibcat_star[1] * 0.01745329252
        separation_arcsec = 206264.806247*(float(ephem.separation(
            (calibcat_star_ra_rad, calibcat_star_dec_rad), 
            (sexcat_star_ra_rad, sexcat_star_dec_rad))))
        if separation_arcsec < 5:
            combined_starlist.append([calibcat_star[0], calibcat_star[1], 
                calibcat_star[2], calibcat_star[3], 
                sexcat_star[2], sexcat_star[3], 
                sexcat_star[4]])
# Use the combined_starlist to calculate a zeropoint for the science image.
zeropoint_list = []
zeropoint_err_list = []
for star in combined_starlist:
    ra = star[0]
    dec = star[1]
    tmass_mag = star[2]
    tmass_e_mag = star[3]
    ptel_mag = star[4]
    ptel_e_mag = star[5]
    ptel_flag = star[6]
    star_ra_rad = ra * 0.01745329252
    star_dec_rad = dec * 0.01745329252
    target_separation_arcsec = 206264.806247*(float(ephem.separation(
            (target_ra_rad, target_dec_rad), 
            (star_ra_rad, star_dec_rad))))
    if ((target_separation_arcsec > 5) and 
        ((ptel_flag == 0) or (ptel_flag == 2))):
        zeropoint_list.append(tmass_mag - ptel_mag)
        zeropoint_err_list.append(sqrt(tmass_e_mag*tmass_e_mag + 
            ptel_e_mag*ptel_e_mag))
zeropoint = average(zeropoint_list)
zeropoint_error = average(zeropoint_err_list)
# Now apply the zeropoint to the instrumental magnitudes and create the 
# final_starlist. Store the target photometry in target_mag and target_e_mag.
final_starlist = []
for star in combined_starlist:
    # If the star is our target . . .
    if (206264.806247*(float(ephem.separation(
        (target_ra_rad, target_dec_rad), 
        (star[0] * 0.01745329252, star[1] * 0.01745329252)))) < 5 and
        ((star[6] == 0) or (star[6] == 2) or (star[6] == 4) or (star[6] == 6) or 
        (star[6] == 7))):
        ra = star[0]
        dec = star[1]
        tmass_mag = star[2]
        tmass_e_mag = star[3]
        ptel_mag = star[4]
        ptel_e_mag = star[5]
        ptel_flag = star[6]
        new_mag = ptel_mag + zeropoint
        new_e_mag = sqrt(zeropoint_error*zeropoint_error + 
            ptel_e_mag*ptel_e_mag)
        target_mag = new_mag
        target_e_mag = new_e_mag
        final_starlist.append([ra, dec, tmass_mag, tmass_e_mag, 
            ptel_mag, ptel_e_mag, ptel_flag, new_mag, new_e_mag])
        continue
    # If the star is just a field 2MASS star . . .
    if star[6] == 0:
        ra = star[0]
        dec = star[1]
        tmass_mag = star[2]
        tmass_e_mag = star[3]
        ptel_mag = star[4]
        ptel_e_mag = star[5]
        ptel_flag = star[6]
        new_mag = ptel_mag + zeropoint
        new_e_mag = sqrt(zeropoint_error*zeropoint_error + 
            ptel_e_mag*ptel_e_mag)
        final_starlist.append([ra, dec, tmass_mag, tmass_e_mag, 
            ptel_mag, ptel_e_mag, ptel_flag, new_mag, new_e_mag])
# Calculate the midpoint heliocentric julian date of the exposure. We use a 
# try/except clause in case something fails and use a placeholder hjd in that
# instance.
try:
    start_time = ephem.date(strt_cpu.replace("-", "/"))
    stop_time = ephem.date(stop_cpu.replace("-", "/"))
    hjd_start_time = heliocentric_julian_date(start_time, target_ra_rad, 
        target_dec_rad)
    hjd_stop_time = heliocentric_julian_date(stop_time, target_ra_rad, 
        target_dec_rad)
    hjd = (hjd_stop_time + hjd_start_time)/2
except:
    hjd = 999
# Write out a final catalog of photometry data. This is useful for diagnostic
# purposes to make sure that the newly observed magntidues are in line with the
# archival 2MASS values. The target's archival data is replaced with the 
# placeholder values (999).
abs_2mass_deviation_list = []
output_file = file(sexcat.replace("sex", "finalcat") + ".txt", "w")
output_region_file = file(sexcat.replace("sex", "finalcat") + ".reg", "w")
output_file.write(str(hjd) + "\nRA\t\t\tDEC\t\t\t\t2mass_mag\t\t2mass_e_mag\t\tnew_mag" + 
    "\t\t\tnew_e_mag\n")
output_region_file.write("global color=blue dashlist=8 3 width=1 " + 
    "font='helvetica 12 bold' select=1 highlite=1 dash=0 fixed=0 edit=1 " + 
    "move=1 delete=1 include=1 source=1\nfk5\n")
for star in final_starlist:
    output_file.write(("%f" + "\t" + "%f" + "\t\t" + 
        "%f" + "\t\t" + "%f" + "\t\t" + "%f" + "\t\t" + "%f" + "\n") % (star[0], 
        star[1], star[2], star[3], star[7], star[8]))
    if star[2] != 999:
        output_region_file.write(('''circle(%f,%f,4") # color=blue ''' + 
            "width=1 text={%.3f +/-  %.4f}\n") % (star[0], star[1], star[7], 
            star[8]))
        output_region_file.write(("#text(%f,%f) text={2MASS = %.3f +/- %.3f}\n")
            % (star[0], star[1]-0.002222, star[2], star[3]))
        abs_2mass_deviation_list.append(abs(star[2]-star[7]))
    else:
        output_region_file.write(('''circle(%f,%f,4") # color=green ''' + 
            "width=1 text={%.3f +/-  %.4f}\n") % (star[0], star[1], star[7], 
            star[8]))
        output_region_file.write("circle(" + str(star[0])+"," + str(star[1]) + 
            ''',6") # color=cyan width=2 text={TARGET}\n''')
output_file.close()
output_region_file.close()
# Form the photometry string which will be printed out as the final result.
photometry_string = "Source Extractor detects no source at target position."
if str(target_mag) != "999":
    photometry_string = ("%s_mag: %.3f \terr: %f" % 
        (band_type, target_mag, target_e_mag))
# Print out photometry data. 
print progenitor_image_name, "HJD:", hjd
print "Photometry Results:", photometry_string
print ("2MASS Catalog comparison average absolute deviation: " + 
    str(average(abs_2mass_deviation_list)))
print "Zeropoint:", zeropoint, "err", zeropoint_error
# Sort the instrumental magnitudes to find the faintest detection.
sex_inst_mag_list_sorted = sorted(sex_inst_mag_list, key=operator.itemgetter(1))
faintest_mag = sex_inst_mag_list_sorted[-1][0]
faintest_mag_err = sex_inst_mag_list_sorted[-1][1]
faintest_ra = sex_inst_mag_list_sorted[-1][2]
faintest_dec = sex_inst_mag_list_sorted[-1][3]
print ("SExtractor faintest detection: " + str(faintest_mag + zeropoint) + 
    " err " + str(sqrt(faintest_mag_err**2 + zeropoint_error**2)) + " at " + 
    str(faintest_ra) + ", " + str(faintest_dec))
# Clean up the photometry catalog files.
# system("rm viz_output.txt")
# system("rm viz_output_cropped.txt")
# system("rm " + progenitor_image_name.replace(".fits", ".sex"))
# system("rm " + weight_image_name.replace(".fits", ".sex"))