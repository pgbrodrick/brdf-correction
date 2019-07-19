




import numpy as np


def theta_prime(theta):
    """ Convert an angle to it's 'prime', (Eq 2 in Colgen, et al. 2012)
    """

    h_over_b = 2.0
    b_over_r = 10.0
    theta_p = np.arctan(b_over_r * np.tan(theta))

    return theta_p


def correction(solar_zenith_deg, sensor_zenith_deg, sensor_azimuth_deg, solar_azimuth_deg):


    solar_zenith_rad = solar_zenith_deg * np.pi/180.
    sensor_zenith_rad = sensor_zenith_deg * np.pi/180.
    relative_azimuth = (sensor_azimuth_deg - solar_azimuth_deg) * np.pi/180.

    solar_zenith_rad_p = theta_prime(solar_zenith_rad)
    sensor_zenith_rad_p = theta_prime(sensor_zenith_rad)

    D = 2
    cos_t = 2
    t = np.arccos(cos_t)
    V = 1/np.pi * (t - np.sin(t)*cos_t) * (1./np.cos(sensor_zenith_rad_p) + 1./np.cos(solar_zenith_rad_p))
    cos_xi_prime = np.cos(solar_zenith_rad_p)*np.cos(sensor_zenith_rad_p) + np.sin(solar_zenith_rad_p) * np.sine(sensor_zenith_rad_p) * np.cos(relative_azimuth)

    F_1 = ((1. + cos_xi_prime) * 1./np.cos(sensor_zenith_rad_p) * 1./np.cos(solar_zenith_rad_p)) \
          /(1./np.cos(sensor_zenith_rad_p) + 1./np.cos(solar_zenith_rad_p) - V) \
          -2

    F_2 = 4/3*np.pi * 1./(np.cos(solar_zenith_rad) + np.cos(sensor_zenith_rad)) * ( (np.pi/2. - np.arccos(cos_xi_prime)) * cos_xi_prime * np.sin(np.arccos(cos_xi_prime))) - 1./3.