
import numpy as np
import argparse
import gdal
from tqdm import tqdm
import os
import brdf_correction


np.random.seed(13)

parser = argparse.ArgumentParser(description='Perform BRDF correction on mosaic')
parser.add_argument('reflectance_file')
parser.add_argument('obs_file')
parser.add_argument('output_file')
args = parser.parse_args()

refl_set = gdal.Open(args.reflectance_file,gdal.GA_ReadOnly)
obs_set = gdal.Open(args.obs_file,gdal.GA_ReadOnly)


# Generate coefficient table
base = '/lustre/scratch/pbrodrick/colorado/brdf_correction/munged_2/'
num_samples=100000
refl = np.memmap(os.path.join(base,'brdf_training_refl.npy'),mode='r',shape=(num_samples,426),dtype=np.float32)
good_dat = np.logical_not(np.all(refl == 0,axis=1))
refl = refl[good_dat,:]
tch  = np.memmap(os.path.join(base,'brdf_training_tch.npy'),mode='r',shape=(num_samples,1),dtype=np.float32)
shade= np.memmap(os.path.join(base,'brdf_training_shade.npy'),mode='r',shape=(num_samples,1),dtype=np.float32)
obs = np.memmap(os.path.join(base,'brdf_training_obs.npy'),mode='r',shape=(num_samples,10),dtype=np.float32)

tch = tch[good_dat,0]
shade = shade[good_dat,0]
obs = obs[good_dat,:]
coeff_mat = brdf_correction.generate_coeff_table(refl[shade == 1,:], tch[shade == 1], shade[shade == 1], (obs[shade==1,:])[:,[4,2,1,3]])


driver = gdal.GetDriverByName('ENVI')
driver.Register()
outDataset = driver.Create(args.output_file,refl_set.RasterXSize,refl_set.RasterYSize,refl_set.RasterCount,gdal.GDT_Float32,options=['INTERLEAVE=BIL'])
outDataset.SetGeoTransform(refl_set.GetGeoTransform())
outDataset.SetProjection(refl_set.GetProjection())
outDataset.SetMetadata(refl_set.GetMetadata())
del outDataset


# Step 2 - Apply the reference table
for _row in tqdm(range(refl_set.RasterYSize),ncols=80):
    loc_refl = np.transpose(np.squeeze(refl_set.ReadAsArray(0,_row,refl_set.RasterXSize,1)))
    loc_obs = np.transpose(np.squeeze(obs_set.ReadAsArray(0,_row,obs_set.RasterXSize,1)))
        
    rev_refl = brdf_correction.apply_brdf_model(coeff_mat, loc_refl, loc_obs[:,[4,2,1,3]], np.ones(loc_refl.shape[0]), np.ones(loc_refl.shape[0]))
    # x, b

    rev_refl = np.transpose(rev_refl)
    rev_refl = np.reshape(rev_refl, (1, rev_refl.shape[0],rev_refl.shape[1]))
    # y, b, x

    refl_memmap = np.memmap(args.output_file,mode='r+',shape=(refl_set.RasterYSize,refl_set.RasterCount,refl_set.RasterXSize),dtype=np.float32)
    refl_memmap[_row:_row+1,...] = rev_refl
    del refl_memmap



