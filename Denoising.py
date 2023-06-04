# **** imports ****
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float, color, io
from skimage.restoration import (   denoise_tv_chambolle, 
                                    denoise_bilateral,
                                    denoise_wavelet,
                                    estimate_sigma)
from skimage.util import random_noise


# **** read kitten image ****
kitten = img_as_float(io.imread('./images/pexels-kitten.jpg'))
#kitten = img_as_float(io.imread('./images/mn_plate.jpg'))

# **** show kitten image ****
plt.figure(figsize=(10, 8))
plt.imshow(kitten)
plt.title('Kitten')
plt.show()


# **** add noise to kitten image ****
sigma = 0.155                                           # noise standard deviation  was: 0.155
noisy_kitten = random_noise(kitten,
                            var=sigma**2)

# **** show noisy kitten image ****
plt.figure(figsize=(10, 8))
plt.imshow(noisy_kitten)
plt.title('Noisy Kitten')
plt.show()


# **** estimate the noise standard deviation from the noisy image ****
sigma_est = estimate_sigma( noisy_kitten,

                            #multichannel=True,         # deprecated
                            channel_axis=-1,            # for color image

                            average_sigmas=True)

# **** print estimated noise standard deviation 
#      close to the 0.155 standard deviation of the noise we added ****
print(f'Estimated Gaussian noise standard deviation: {sigma_est}')


# **** denoise noisy image using total variation 
#      the greater the weight, more denoising of the input -
#      fidelity of the result might be lower at higher weights ****
denoise_tv_1 = denoise_tv_chambolle(noisy_kitten,
                                    weight=0.1,
                                    
                                    #multichannel=True) # deprecated
                                    channel_axis=-1)    # for color image

# *** denoise with different weight ****
denoise_tv_2 = denoise_tv_chambolle(noisy_kitten,
                                    weight=0.2,
                                    channel_axis=-1)

# **** show denoised images ****
fig, ax = plt.subplots( nrows=1,
                        ncols=2,
                        figsize=(18, 14),
                        sharex=True,
                        sharey=True)

ax[0].imshow(denoise_tv_1)
ax[0].axis('off')
ax[0].set_title('Total Variation (TV) Denoising (Weight = 0.1)')

ax[1].imshow(denoise_tv_2)
ax[1].axis('off')
ax[1].set_title('Total Variation (TV) Denoising (Weight = 0.2)')

fig.tight_layout()
plt.show()


# **** denoise noisy image using an edge preserving bilateral filter
#      averages pixels based on their spatial closeness and 
#      radiometric similarity (how close the colors are) ****
denoise_bi_1 = denoise_bilateral(   noisy_kitten,
                                    sigma_color=0.05,
                                    sigma_spatial=15,
                                    
                                    #multichannel=True)
                                    channel_axis=-1)

# **** denoise with different sigma_color
#      accept a larger standard deviation while averaging pixels based on color ****
denoise_bi_2 = denoise_bilateral(   noisy_kitten,
                                    sigma_color=0.1,
                                    sigma_spatial=15,
                                    channel_axis=-1)

# **** show denoised images ****
fig, ax = plt.subplots( nrows=1,
                        ncols=2,
                        figsize=(18, 14),
                        sharex=True,
                        sharey=True)

ax[0].imshow(denoise_bi_1)
ax[0].axis('off')
ax[0].set_title('Bilateral Denoising (Sigma Color = 0.05)')

ax[1].imshow(denoise_bi_2)
ax[1].axis('off')
ax[1].set_title('Bilateral Denoising (Sigma Color = 0.1)')

fig.tight_layout()
plt.show()


# **** denoise noisy image using wavelet denoising 
#      soft thresholding gives a better approximation of the original image
#      denoising images is better with the YCbCr color space 
#      as compared to the RGB color space ****
denoise_wave_1 = denoise_wavelet(   noisy_kitten,
                                    mode='soft',
                                    
                                    #multichannel=True,
                                    channel_axis=-1,

                                    convert2ycbcr=True)

# **** denoise keeping image in RGB space ****
denoise_wave_2 = denoise_wavelet(   noisy_kitten,
                                    mode='soft',
                                    channel_axis=-1,
                                    convert2ycbcr=False)

# **** show denoised images ****
fig, ax = plt.subplots( nrows=1,
                        ncols=2,
                        figsize=(18, 14),
                        sharex=True,
                        sharey=True)

ax[0].imshow(denoise_wave_1)
ax[0].axis('off')
ax[0].set_title('Wavelet Denoising (YCbCr)')

ax[1].imshow(denoise_wave_2)
ax[1].axis('off')
ax[1].set_title('Wavelet Denoising (RGB)')

fig.tight_layout()
plt.show()
