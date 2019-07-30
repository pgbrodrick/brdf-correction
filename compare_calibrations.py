



import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


import brdf_correction
import numpy as np

num_samples = 100000
refl = np.memmap('munged/brdf_training_refl.npy',mode='r',shape=(num_samples,426),dtype=np.float32)
good_dat = np.logical_not(np.all(refl == 0,axis=1))
refl = refl[good_dat,:]
tch  = np.memmap('munged/brdf_training_tch.npy',mode='r',shape=(num_samples,1),dtype=np.float32)
shade= np.memmap('munged/brdf_training_shade.npy',mode='r',shape=(num_samples,1),dtype=np.float32)
obs = np.memmap('munged/brdf_training_obs.npy',mode='r',shape=(num_samples,10),dtype=np.float32)

tch = tch[good_dat,:]
shade = shade[good_dat,:]
obs = obs[good_dat,:]

np.random.seed(13)
perm = np.random.permutation(refl.shape[0])
refl = refl[perm,:]
tch = tch[perm,:]
shade = shade[perm,:]
obs = obs[perm,:]


perm = np.random.permutation(refl.shape[0])
num_folds = 3
coeff_mat = []
shade_coeff_mat = []
combo_coeff_mat = []
for _fold in range(num_folds):
    subset = perm[int(_fold * refl.shape[0] / num_folds):int((_fold+1) * refl.shape[0] / num_folds)]
    refl_subset = refl[subset,:]
    obs_subset = obs[subset,:]
    tch_subset = np.squeeze(tch[subset])
    shade_subset = np.squeeze(shade[subset])
    
    coeff_mat.append(brdf_correction.generate_coeff_table(refl_subset[shade_subset == 1,:], 
                                                          tch_subset[shade_subset == 1], 
                                                          shade_subset[shade_subset == 1], 
                                                          (obs_subset[shade_subset==1,:])[:,[4,2,1,3]]))

    shade_coeff_mat.append(brdf_correction.generate_coeff_table(refl_subset[shade_subset == 0,:], 
                                                          tch_subset[shade_subset == 0], 
                                                          shade_subset[shade_subset == 0], 
                                                          (obs_subset[shade_subset==0,:])[:,[4,2,1,3]]))

    combo_coeff_mat.append(brdf_correction.generate_coeff_table(refl_subset, 
                                                          tch_subset, 
                                                          shade_subset, 
                                                          obs_subset[:,[4,2,1,3]]))


coeff_mat = np.stack(coeff_mat)
shade_coeff_mat = np.stack(shade_coeff_mat)
combo_coeff_mat = np.stack(combo_coeff_mat)


def find_nearest(array_like, v):
    return np.argmin(np.abs(np.array(array_like) - v))
    

wv = np.array([383.68 + i*5 for i in range(426)])
bad_bands = []
for n in range(10): bad_bands.append(n)
for n in range(find_nearest(wv,1355),find_nearest(wv,1400)): bad_bands.append(n)
for n in range(find_nearest(wv,1815),find_nearest(wv,2000)): bad_bands.append(n)
for n in range(find_nearest(wv,2480),426): bad_bands.append(n)
good_bands = np.array([x for x in range(0,426) if x not in bad_bands])

un_class = coeff_mat.shape[1]

coeff_mat[:,:,bad_bands,:] = np.nan
shade_coeff_mat[:,:,bad_bands,:] = np.nan
combo_coeff_mat[:,:,bad_bands,:] = np.nan


fig = plt.figure(figsize=(8,8))

for _c in range(3):
    ax = plt.subplot2grid((3, 1), (_c, 0))
    for n in range(num_folds):
        plt.plot(wv, np.squeeze(coeff_mat[n,:,:,_c]),c='green')
        plt.plot(wv, np.squeeze(shade_coeff_mat[n,:,:,_c]),c='red')
        plt.plot(wv, np.squeeze(combo_coeff_mat[n,:,:,_c]),c='blue')


plt.savefig('figs/coeff_comparison.png',dpi=200)




