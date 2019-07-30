



import pandas as pd
import numpy as np
import gdal


from tqdm import tqdm


refl_file = '../mosaic_2019/built_mosaic/min_phase_refl'
tch_file = '../mosaic_2019/built_mosaic/min_phase_tch_me'
shade_file = '../mosaic_2019/built_mosaic/min_phase_shade'
obs_file = '../mosaic_2019/built_mosaic/min_phase_obs'

refl_set = gdal.Open(refl_file,gdal.GA_ReadOnly)
tch_set = gdal.Open(tch_file,gdal.GA_ReadOnly)
shade_set = gdal.Open(shade_file,gdal.GA_ReadOnly)
obs_set = gdal.Open(obs_file,gdal.GA_ReadOnly)
all_sets = [refl_set,tch_set,shade_set,obs_set]

for i in range(1,len(all_sets)):
    assert all_sets[0].GetGeoTransform() == all_sets[i].GetGeoTransform(), 'geotrans match fail'
    #assert all_sets[0].GetProjection() == all_sets[i].GetProjection(), 'proj match fail'
    assert all_sets[0].RasterXSize == all_sets[i].RasterXSize, 'raster x size fail'
    assert all_sets[0].RasterYSize == all_sets[i].RasterYSize, 'raster y size fail'

x_len = tch_set.RasterXSize
y_len = tch_set.RasterYSize

np.random.seed(13)
num_samples = 100000
sample_loc = np.random.permutation(y_len*x_len)
sample_loc = sample_loc.reshape((y_len,x_len)) 
sample_loc = (sample_loc < num_samples*3).astype(bool)
y_px, x_px = np.where(sample_loc)
del sample_loc
perm = np.random.permutation(len(y_px))
y_px = y_px[perm]
x_px = x_px[perm]
del perm

all_refl = []
all_tch = []
all_shade = []
all_obs = []

refl_outfile = np.memmap
def get_files(style):
    rf = np.memmap('munged_2/brdf_training_refl.npy',mode=style,shape=(num_samples,426),dtype=np.float32)
    tf = np.memmap('munged_2/brdf_training_tch.npy',mode=style,shape=(num_samples,1),dtype=np.float32)
    sf = np.memmap('munged_2/brdf_training_shade.npy',mode=style,shape=(num_samples,1),dtype=np.float32)
    of = np.memmap('munged_2/brdf_training_obs.npy',mode=style,shape=(num_samples,10),dtype=np.float32)
    return rf, tf, sf, of

rf, tf, sf, of = get_files('w+')
del rf, tf, sf, of
ind = 0

for _i in tqdm(range(len(y_px)), ncols=80, desc='Reading Spectral Files'):

    refl_line = refl_set.ReadAsArray(0,int(y_px[_i]),refl_set.RasterXSize,1)
    refl_dat = refl_line[:,:,int(x_px[_i])]
    if (np.all(refl_dat == 0) == False):

        tch_dat =     tch_set.ReadAsArray(int(x_px[_i]), int(y_px[_i]), 1, 1)
        shade_dat = shade_set.ReadAsArray(int(x_px[_i]), int(y_px[_i]), 1, 1)

        obs_line = obs_set.ReadAsArray(0,int(y_px[_i]),obs_set.RasterXSize,1)
        obs_dat = obs_line[:,:,int(x_px[_i])]
        
        rf, tf, sf, of = get_files('r+')
        rf[ind,:] = np.squeeze(refl_dat)
        tf[ind] = tch_dat
        sf[ind] = shade_dat
        of[ind,:] = np.squeeze(obs_dat)
        ind += 1

        del rf, tf, sf, of
        del refl_dat, tch_dat, shade_dat, obs_dat, refl_line, obs_line

    if(len(all_refl) >= num_samples):
        break









