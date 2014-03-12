import pyfits





header_keys_convert_to_floats = [
    "EQUINOX",
    "CRVAL1",
    "CRPIX1",
    "CD1_1",
    "CD1_2",
    "CRVAL2",
    "CRPIX2",
    "CD2_1",
    "CD2_2",
    "EXPTIME",
    "GAIN",
    "SATURATE",
    "WCSAXES",
    "LONPOLE",
    "LATPOLE",
    "IMAGEW",
    "IMAGEH",
    "A_ORDER",
    "A_0_2",
    "A_1_1",
    "A_2_0",
    "B_ORDER",
    "B_0_2",
    "B_1_1",
    "B_2_0",
    "AP_ORDER",
    "AP_0_1",
    "AP_0_2",
    "AP_1_0",
    "AP_1_1",
    "AP_2_0",
    "BP_ORDER",
    "BP_0_1",
    "BP_0_2",
    "BP_1_0",
    "BP_1_1",
    "BP_2_0",
    ]

input_image = "h_long_RATS.40.146_coadd.weight.fits"

input_hdu = pyfits.open(input_image)

image_data = input_hdu[0].data

converted_imagefile_header = input_hdu[0].header.copy()

for hk in header_keys_convert_to_floats:
    try:
        converted_imagefile_header.update(hk, float(input_hdu[0].header[hk]))
    except:
        print hk, "header update failed"


output_hdu = pyfits.PrimaryHDU(image_data)
output_hdu.header = converted_imagefile_header


output_hdu.verify("fix")
output_hdulist = pyfits.HDUList([output_hdu])
output_hdulist.writeto(input_image.replace(".fits", ".convertedwcs.fits"))

input_hdu.close()