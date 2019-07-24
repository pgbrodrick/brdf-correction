




import numpy as np

h_over_b = 2.0
b_over_r = 10.0
n_classes = 4
n_bands = 426

def theta_prime(theta):
    """ Convert an angle to it's 'prime', (Eq 2 in Colgen, et al. 2012)
    """

    theta_p = np.arctan(b_over_r * np.tan(theta))

    return theta_p


def correction_components(solar_zenith_deg, sensor_zenith_deg, sensor_azimuth_deg, solar_azimuth_deg):

    solar_zenith_rad = solar_zenith_deg * np.pi/180.
    sensor_zenith_rad = sensor_zenith_deg * np.pi/180.
    relative_azimuth = (sensor_azimuth_deg - solar_azimuth_deg) * np.pi/180.

    solar_zenith_rad_p = theta_prime(solar_zenith_rad)
    sensor_zenith_rad_p = theta_prime(sensor_zenith_rad)

    D = np.sqrt(np.power(np.tan(solar_zenith_rad_p),2) + np.power(np.tan(sensor_zenith_rad_p),2) - 2*np.tan(sensor_zenith_rad_p)*np.tan(solar_zenith_rad_p)*np.cos(relative_azimuth))
    cos_t = h_over_b * np.sqrt(np.power(D,2) + np.tan(solar_zenith_rad_p) * np.tan(sensor_zenith_rad_p) * np.sin(relative_azimuth)) / (1/np.cos(solar_zenith_rad_p) + 1/np.cos(sensor_zenith_rad_p))
    t = np.arccos(cos_t)
    V = 1/np.pi * (t - np.sin(t)*cos_t) * (1./np.cos(sensor_zenith_rad_p) + 1./np.cos(solar_zenith_rad_p))
    cos_xi_prime = np.cos(solar_zenith_rad_p)*np.cos(sensor_zenith_rad_p) + np.sin(solar_zenith_rad_p) * np.sine(sensor_zenith_rad_p) * np.cos(relative_azimuth)

    F_1 = ((1. + cos_xi_prime) * 1./np.cos(sensor_zenith_rad_p) * 1./np.cos(solar_zenith_rad_p)) \
          /(1./np.cos(sensor_zenith_rad_p) + 1./np.cos(solar_zenith_rad_p) - V) \
          -2

    F_2 = 4/3*np.pi * 1./(np.cos(solar_zenith_rad) + np.cos(sensor_zenith_rad)) * ( (np.pi/2. - np.arccos(cos_xi_prime)) * cos_xi_prime * np.sin(np.arccos(cos_xi_prime))) - 1./3.

    return F_1, F_2

def generate_coeff_table(refl, tch, shade, relobs):
    roughclass = separate_classes(getndvi(refl), tch, shade)
    coeff_mat = np.zeros((n_classes, n_bands, 3))

    for _class in range(n_classes):
        subset = roughclass == _class
        for _band in range(n_bands):
            coeff_mat[_class,_band,:] + calculate_coefficients(refl[subset,_band], relobs)

    return coeff_mat


def calculate_coefficients(refl, relobs):
    # solve the equation c0 + c1 F_1 + c2 F_2 by minimizing least squares
    # A = [1, F_1, F_2], x = [c0,c1,c2]

    F_1, F_2 = correction_components(relobs[:,0], relobs[:,1], relobs[:,2], relobs[:,3])

    A = np.transpose(np.vstack([np.ones(F_1.shape), F_1, F_2]))

    coeff = np.linalg.lstsq(A, refl)
    return coeff


def separate_classes(ndvi, tch, shade):

    outclass = np.zeros(ndvi.shape[0])
    outclass[shade == 0] = 0 #shaded
    outclass[np.logical_and.reduce((tch > 2, shade == 1))] = 1 #sunlit tree
    outclass[np.logical_and.reduce((tch < 2, shade == 1, ndvi > 0.2))] = 2 #sunlit short veg
    outclass[np.logical_and.reduce((tch < 2, shade == 1, ndvi <= 0.2))] = 3 #sunlit short non-veg

    return outclass


def apply_brdf_model(coef, refl, relobs, tch, shade):

    F_1, F_2 = correction_components(relobs[:,0], relobs[:,1], relobs[:,2], relobs[:,3])
    refF_1, refF_2 = correction_components(np.ones(relobs[:,0].shape)*40, np.zeros(relobs[:,0].shape), np.zeros(relobs[:,0].shape), np.zeros(relobs[:,0].shape))
    roughclass = separate_classes(getndvi(refl), tch, shade)


    for _class in range(n_classes):
        subset = roughclass == _class
        for _band in range(n_bands):
            modeled_r = coef[_class,_band,0] + F_1[subset] * coef[_class,_band,1] * F_2[subset]*coef[_class,_band,2]
            ref_r = coef[_class,_band,0] + refF_1[subset] * coef[_class,_band,1] * refF_2[subset]*coef[_class,_band,2]
            refl[subset,_band] = refl[subset,_band] * ref_r / modeled_r


    return refl



















