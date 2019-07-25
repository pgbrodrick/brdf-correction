




import numpy as np
import argparse
import gdal
from tqdm import tqdm


h_over_b = 2.0
b_over_r = 10.0
n_classes = 4
n_bands = 426
wv = np.array([383.68 + i*5 for i in range(426)])


def get_ndvi(refl):
    _860 = np.argmin(np.abs(np.array(wv) - 860))
    _660 = np.argmin(np.abs(np.array(wv) - 860))
    return (refl[:,_860] - refl[:,_660]) / (refl[:,_860] + refl[:,_660])

def theta_prime(theta):
    """ Convert an angle to it's 'prime', (Eq 2 in Colgen, et al. 2012)
    """

    theta_p = np.arctan(b_over_r * np.tan(theta))

    return theta_p


def correction_components(solar_zenith_deg, sensor_zenith_deg, sensor_azimuth_deg, solar_azimuth_deg):

    solar_zenith_rad = solar_zenith_deg * np.pi/180.
    sensor_zenith_rad = sensor_zenith_deg * np.pi/180.
    relative_azimuth = (solar_azimuth_deg - sensor_azimuth_deg) * np.pi/180.

    solar_zenith_rad_p = theta_prime(solar_zenith_rad)
    sensor_zenith_rad_p = theta_prime(sensor_zenith_rad)

    sec_Tv = 1/np.cos(sensor_zenith_rad_p)
    sec_Ts = 1/np.cos(solar_zenith_rad_p)
    tan_Tv = np.tan(sensor_zenith_rad_p)
    tan_Ts = np.tan(solar_zenith_rad_p)
    cos_Tv = np.cos(sensor_zenith_rad_p)
    cos_Ts = np.cos(solar_zenith_rad_p)
    sin_Tv = np.sin(sensor_zenith_rad_p)
    sin_Ts = np.sin(solar_zenith_rad_p)

    cos_p = np.cos(relative_azimuth)
    sin_p = np.sin(relative_azimuth)


    #D = np.sqrt(np.power(np.tan(solar_zenith_rad_p),2) + np.power(np.tan(sensor_zenith_rad_p),2) - 2*np.tan(sensor_zenith_rad_p)*np.tan(solar_zenith_rad_p)*np.cos(relative_azimuth))
    D = np.sqrt(np.power(tan_Ts,2) + np.power(tan_Tv,2) - 2*tan_Ts*tan_Tv*cos_p)

    #cos_t = h_over_b * np.sqrt(np.power(D,2) + np.power(np.tan(solar_zenith_rad_p) * np.tan(sensor_zenith_rad_p) * np.sin(relative_azimuth),2)) / (1/np.cos(solar_zenith_rad_p) + 1/np.cos(sensor_zenith_rad_p))
    cos_t = h_over_b * np.sqrt(np.power(D,2) + np.power(tan_Ts*tan_Tv*sin_p,2)) / (sec_Tv + sec_Ts)
    cos_t = np.min([cos_t, np.ones(cos_t.shape)],axis=0)
    t = np.arccos(cos_t)
    V = 1/np.pi * (t - np.sin(t)*cos_t) * (1./np.cos(sensor_zenith_rad_p) + 1./np.cos(solar_zenith_rad_p))
    #cos_xi_prime = np.cos(solar_zenith_rad_p)*np.cos(sensor_zenith_rad_p) + np.sin(solar_zenith_rad_p) * np.sin(sensor_zenith_rad_p) * np.cos(relative_azimuth)
    cos_xi_prime = cos_Ts*cos_Tv + sin_Ts*sin_Tv*cos_p


    #F_1 = ((1. + cos_xi_prime) * 1./np.cos(sensor_zenith_rad_p) * 1./np.cos(solar_zenith_rad_p)) \
    #      /(1./np.cos(sensor_zenith_rad_p) + 1./np.cos(solar_zenith_rad_p) - V) \
    #      -2
    F_1 = (1+cos_xi_prime)*sec_Tv*sec_Ts / (sec_Tv + sec_Ts - V) - 2

    cos_xi = np.cos(solar_zenith_rad)*np.cos(sensor_zenith_rad) + np.sin(solar_zenith_rad)*np.sin(sensor_zenith_rad)*np.cos(relative_azimuth)
    #F_2 = 4/(3*np.pi) * 1./(np.cos(solar_zenith_rad_p) + np.cos(sensor_zenith_rad_p)) * ( (np.pi/2. - np.arccos(cos_xi_prime)) * cos_xi_prime + np.sin(np.arccos(cos_xi_prime))) - 1./3.
    F_2 = 4/(3*np.pi) * 1./(np.sin(solar_zenith_rad) + np.cos(sensor_zenith_rad)) * ( (np.pi/2. - np.arccos(cos_xi)) * cos_xi + np.sin(np.arccos(cos_xi))) - 1./3.


    return F_1, F_2

def generate_coeff_table(refl, tch, shade, relobs):
    roughclass = separate_classes(get_ndvi(refl), tch, shade)
    coeff_mat = np.zeros((n_classes, n_bands, 3))

    for _class in range(n_classes):
        subset = roughclass == _class
        for _band in range(n_bands):
            coeff_mat[_class,_band,:] = calculate_coefficients(refl[subset,_band], relobs[subset,:])

    return coeff_mat


def calculate_coefficients(refl, relobs):
    # solve the equation c0 + c1 F_1 + c2 F_2 by minimizing least squares
    # A = [1, F_1, F_2], x = [c0,c1,c2]

    F_1, F_2 = correction_components(relobs[:,0], relobs[:,1], relobs[:,2], relobs[:,3])

    A = np.transpose(np.vstack([np.ones(F_1.shape), F_1, F_2]))

    coeff, resid, rank, s = np.linalg.lstsq(A, refl)
    return coeff


def separate_classes(ndvi, tch, shade):

    outclass = np.zeros(ndvi.shape[0])
    outclass[np.logical_and.reduce((tch > 2, shade == 1))] = 0 #sunlit tree
    outclass[np.logical_and.reduce((tch > 2, shade == 0))] = 1 #shaded tree
    outclass[np.logical_and.reduce((tch <= 2, shade == 1))] = 2 #sunlit short veg
    outclass[np.logical_and.reduce((tch <= 2, shade == 0))] = 3 #shaded short non-veg

    return outclass


def apply_brdf_model(coef, refl, relobs, tch, shade):

    F_1, F_2 = correction_components(relobs[:,0], relobs[:,1], relobs[:,2], relobs[:,3])
    refF_1, refF_2 = correction_components(np.ones(relobs[:,0].shape)*40, np.zeros(relobs[:,0].shape), np.zeros(relobs[:,0].shape), np.zeros(relobs[:,0].shape))
    roughclass = separate_classes(get_ndvi(refl), tch, shade)

    for _class in range(n_classes):
        subset = roughclass == _class
        for _band in range(n_bands):
            modeled_r = coef[_class,_band,0] + F_1[subset] * coef[_class,_band,1] + F_2[subset]*coef[_class,_band,2]
            ref_r = coef[_class,_band,0] + refF_1[subset] * coef[_class,_band,1] + refF_2[subset]*coef[_class,_band,2]
            correction = refl[subset,_band] * ref_r / modeled_r
            correction[np.isnan(correction)] = 0
            refl[subset,_band] = correction

    return refl



np.random.seed(13)

parser = argparse.ArgumentParser(description='Perform BRDF correction on mosaic')
parser.add_argument('reflectance_file')
parser.add_argument('obs_file')
parser.add_argument('tch_file')
parser.add_argument('shade_file')
parser.add_argument('output_file')
args = parser.parse_args()


# Step 1 - Build reference table

refl_set = gdal.Open(args.reflectance_file,gdal.GA_ReadOnly)
obs_set = gdal.Open(args.obs_file,gdal.GA_ReadOnly)
tch_set = gdal.Open(args.tch_file,gdal.GA_ReadOnly)
shade_set = gdal.Open(args.shade_file,gdal.GA_ReadOnly)

perm = np.random.permutation((refl_set.RasterYSize*refl_set.RasterXSize)).reshape((refl_set.RasterYSize,refl_set.RasterXSize))
num_samples = 1000
matrix_subset = perm < num_samples
del perm

refl = np.zeros((num_samples, refl_set.RasterCount))
obs = np.zeros((num_samples, obs_set.RasterCount))
tch = np.zeros((num_samples))
shade = np.zeros((num_samples))

_ind = 0
for _row in tqdm(range(refl_set.RasterYSize),ncols=80):
    if (np.sum(matrix_subset[_row,:]) > 0):
        loc_tch = np.transpose(np.squeeze(tch_set.ReadAsArray(0,_row,tch_set.RasterXSize,1))[ matrix_subset[_row,:]])
        loc_refl = np.transpose(np.squeeze(refl_set.ReadAsArray(0,_row,refl_set.RasterXSize,1))[:, matrix_subset[_row,:]])
        loc_shade = np.transpose(np.squeeze(shade_set.ReadAsArray(0,_row,shade_set.RasterXSize,1))[matrix_subset[_row,:]])
        loc_obs = np.transpose(np.squeeze(obs_set.ReadAsArray(0,_row,obs_set.RasterXSize,1))[:, matrix_subset[_row,:]])

        tch[_ind:_ind+len(loc_tch)] = loc_tch
        shade[_ind:_ind+len(loc_tch)] = loc_shade
        refl[_ind:_ind+len(loc_tch),:] = loc_refl
        obs[_ind:_ind+len(loc_tch),:] = loc_obs

        _ind += len(loc_tch)

coeff_mat = generate_coeff_table(refl, tch, shade, obs[:,[4,2,1,3]])
print(coeff_mat)

driver = gdal.GetDriverByName('ENVI')
driver.Register()
outDataset = driver.Create(args.output_file,refl_set.RasterXSize,refl_set.RasterYSize,refl_set.RasterCount,gdal.GDT_Float32,options=['INTERLEAVE=BIL'])
outDataset.SetGeoTransform(refl_set.GetGeoTransform())
outDataset.SetProjection(refl_set.GetProjection())
del outDataset



# Step 2 - Apply the reference table
for _row in tqdm(range(refl_set.RasterYSize),ncols=80):
    loc_tch = np.transpose(np.squeeze(tch_set.ReadAsArray(0,_row,tch_set.RasterXSize,1)))
    loc_refl = np.transpose(np.squeeze(refl_set.ReadAsArray(0,_row,refl_set.RasterXSize,1)))
    loc_shade = np.transpose(np.squeeze(shade_set.ReadAsArray(0,_row,shade_set.RasterXSize,1)))
    loc_obs = np.transpose(np.squeeze(obs_set.ReadAsArray(0,_row,obs_set.RasterXSize,1)))
        
    rev_refl = apply_brdf_model(coeff_mat, loc_refl, loc_obs[:,[4,2,1,3]], loc_tch, loc_shade)
    # x, b

    rev_refl = np.transpose(rev_refl)
    rev_refl = np.reshape(rev_refl, (1, rev_refl.shape[0],rev_refl.shape[1]))
    # y, b, x


    refl_memmap = np.memmap(args.output_file,mode='r+',shape=(refl_set.RasterYSize,refl_set.RasterCount,refl_set.RasterXSize),dtype=np.float32)
    refl_memmap[_row:_row+1,...] = rev_refl
    del refl_memmap



