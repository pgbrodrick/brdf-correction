# brdf-correction
Easy to use, python implementation of a brdf correction to hyperspectral data.

Based on Colgen, et al. 2012 (https://doi.org/10.3390/rs4113462).  Li-dense + Ross-thick models.





Ross-thin (LAI << 1). Thus, the modeled surface reflectance R was computed for each band as:
R(θs,θv,φ)=c0+c1F1(θs, θv, φ)+c2F2(θs, θv, φ)  (1)
where θs is the solar zenith angle (radians), θv is the view zenith angle (radians), φ is the relative azimuth angle (radians), F1 is the Li-dense kernel, F2 is the Ross-thick kernel, and ci are constants inverted from measurements, discussed below (where c0 represents the isotropic reflectance). The kernels were computed from [24,37] (with the reciprocal modification to the Li kernel) as:

F1=(1+cos ξ′) sec θ′v sec θ′ssec θ′v+sec θ′s−V−2

V=1π(t−sin t cos t) (sec θ′v+sec θ′s)

cos t=hb D2+(tan θ′s tan θ′v sin φ√sec θ′v+sec θ′s

D=tan2 θ′s+tan2 θ′v−2 tan θ′s tan θ′v cos φ−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−√

cos ξ′=cos θ′s cos θ′v+sin θ′s sin θ′v cos φ

θ′=tan−1(brtan θ)

F2=43π 1cos θs+cos θv [(π2−ξ) cos ξ+sin ξ]−13  (2)

The values for the parameters h/b and b/r (object shape and height) set to 2.0 and 10.0 after iterative testing demonstrated these values minimized the residual difference between test and predicted spectra. A multiplicative model was used to compute the final BRDF-corrected reflectance, ρc, by multiplying the observed reflectance, ρ, by the ratio of the modeled reflectance at the reference geometry (i.e., nadir view and roughly the mean solar zenith angle across all flight lines of 40°) to the modeled reflectance at the observed geometry:

ρc=ρ R(θs=40°, θv=0, φ=0) / R(θs,obs, θv,obs,φobs) (3)

The BRDF model (Equation (1)) constants c1–c3 were calibrated for each reflectance mosaic using several 


