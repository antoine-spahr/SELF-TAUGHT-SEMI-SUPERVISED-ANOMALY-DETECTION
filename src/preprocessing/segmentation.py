import matplotlib.pyplot as plt
import matplotlib
import skimage.draw
import skimage.morphology
import skimage
import numpy as np
import shapely.geometry
import pandas as pd
import scipy.spatial.distance as dist
import PIL.Image
import PIL.ImageDraw

def segment2(img, factors=(0.9,1)):
    """
    Segment the X-ray body part of the image using a Yen Thresholding segmented
    with an hysteresis threshold. The results is then morphologically openened
    with a disk of radius 10. The coutour of the binary mask are extracted, the
    largest one is kept and holes are removed to provide the final mask.
    ----------
    INPUT
        |---- img (np.array 2D or 3D) the image to segement. It can be either a
        |           Greyscale or Color image.
        |---- factors (tuple (low, high)) the hystersis thresholds used in the
        |           scikit-image function 'apply_hysteresis_threshold'.
    OUTPUT
        |---- mask (np.array 2D) the segmentation binary mask.
    """
    if img.ndim == 3 : img = skimage.color.rgb2gray(img)
    pad_val = 5
    img = skimage.exposure.equalize_hist(img)
    thrs = skimage.filters.threshold_yen(img)
    mask = skimage.filters.apply_hysteresis_threshold(img, thrs*factors[0], thrs*factors[1])
    mask = skimage.morphology.opening(mask, selem=skimage.morphology.disk(10))
    # pad to avoid boundary effect
    mask = skimage.util.pad(mask, pad_width=pad_val)
    # get polygons
    poly = skimage.measure.find_contours(mask, 0.5)
    if len(poly) > 0:
        # remove padding
        poly = [p-pad_val for p in poly]
        # Keep only larger polygon
        poly = [shapely.geometry.Polygon(p) for p in poly]
        main_poly = max(poly, key=lambda a: a.area)

        mask = np.zeros_like(img, np.uint8)
        rc = np.array(list(main_poly.exterior.coords))
        rr , cc = skimage.draw.polygon(rc[:,0], rc[:,1])
        mask[rr, cc] = 1
        #poly.remove(main_poly)
        for p in [p for p in poly if p.area < main_poly.area]:
            if main_poly.contains(p):
                rc = np.array(list(p.exterior.coords))
                rr, cc = skimage.draw.polygon(rc[:,0], rc[:,1])
                mask[rr, cc] = 0
    else:
        mask = mask[pad_val:-pad_val, pad_val:-pad_val]

    return mask

def segment(img, factors=(0.9,1)):
    """
    Segment the X-ray body part of the image using a Yen Thresholding segmented
    with an hysteresis threshold. The results is then morphologically openened
    with a disk of radius 10. The coutour of the binary mask are extracted, the
    largest one is kept and holes are removed to provide the final mask.
    ----------
    INPUT
        |---- img (np.array 2D or 3D) the image to segement. It can be either a
        |           Greyscale or Color image.
        |---- factors (tuple (low, high)) the hystersis thresholds used in the
        |           scikit-image function 'apply_hysteresis_threshold'.
    OUTPUT
        |---- mask (np.array 2D) the segmentation binary mask.
    """
    if img.ndim == 3 : img = skimage.color.rgb2gray(img)
    pad_val = 5
    img = skimage.exposure.equalize_hist(img)
    thrs = skimage.filters.threshold_yen(img)
    mask = skimage.filters.apply_hysteresis_threshold(img, thrs*factors[0], thrs*factors[1])
    mask = skimage.morphology.binary_opening(mask, selem=skimage.morphology.disk(10))
    # pad to avoid boundary effect
    mask = skimage.util.pad(mask, pad_width=pad_val)
    # get polygons
    poly = skimage.measure.find_contours(mask, 0.5)
    if len(poly) > 0:
        # remove padding
        poly = [p-pad_val for p in poly]
        # Keep only larger polygon
        poly = [shapely.geometry.Polygon(p) for p in poly]
        main_poly = max(poly, key=lambda a: a.area)
        # get holes
        poly = [p for p in poly if (main_poly.contains(p)) and (p.area < main_poly.area)]
        # draw mask
        mask = PIL.Image.new("1", img.shape[::-1], 0)
        PIL.ImageDraw.Draw(mask).polygon([c[::-1] for c in list(main_poly.exterior.coords)], outline=1, fill=1)
        for p in poly:
            PIL.ImageDraw.Draw(mask).polygon([c[::-1] for c in list(p.exterior.coords)], outline=0, fill=0)
        mask = np.array(mask)
    else:
        mask = mask[pad_val:-pad_val, pad_val:-pad_val]

    return mask

def find_best_mask(img, tol_factor=0.1, search_res=0.1, search_range=(0.2,1)):
    """
    Find the best masks over different thresholding factors. The best mask is the
    one minimizing the correlation distance between the image and the computed mask.
    To keep a lower 'low' threshold as much as possible, a tolerance on the minimum
    correlation is imposed : the new correlation (for higher threhold) must be
    smaller than the minimum correlation distance minus tol_factor% of the current range.
    ----------
    INPUT
        |---- img (np.array 2D or 3D) the image to segement. It can be either a
        |           Greyscale or Color image.
        |---- tol_factor (float) the tolerance on the minimum correlation distance
        |---- search_res (float) the resolution of the factor search
        |---- search range (tuple (lower, upper)) the range over which the factor
        |           search is done
    OUTPUT
        |---- best_mask (np.array 2D) the best segmentation binary mask.
    """
    if img.ndim == 3 : img = skimage.color.rgb2gray(img)
    min_corr = np.inf
    max_corr = -np.inf
    best_mask = np.ones_like(img)
    for i in np.arange(search_range[0], search_range[1]+0.01, search_res):
        mask = segment(img, factors=(i,1))
        # check if mask is constant
        if mask.min() != mask.max():
            corr = dist.correlation(img.flatten(), mask.flatten())
        else:
            corr = np.nan
        # upadte if new correlation distance is lower
        if corr > max_corr :
            max_corr = corr
        if corr < (min_corr - tol_factor * (max_corr - corr)): # tolerance on minimum
            min_corr = corr
            best_mask = mask
    return best_mask





# %% ###########################################################################
################################################################################
################################################################################

# IN_DATA_PATH = '../../../data/RAW/'
# OUT_DATA_PATH = '../../../data/PROCESSED/'
# DATAINFO_PATH = '../../../data/'
#
# df = pd.read_csv(DATAINFO_PATH + 'data_info.csv')
# fn = df.loc[df.body_part == 'HAND', 'filename']
# img = skimage.io.imread(IN_DATA_PATH + fn.iloc[589])
# plt.imshow(img, cmap='Greys_r')
# #img = skimage.io.imread(IN_DATA_PATH + 'XR_HUMERUS/patient00119/study1_negative/image1.png')
# # invert image if negatives (i.e. if too bright)
# if img.mean() > 125:
#     img = np.invert(img)
#
# best_mask = find_best_mask(img, tol_factor=0.1, \
#                            search_res=0.1, search_range=(0.2,1))
#
# # %%
# if img.ndim == 3 : img = skimage.color.rgb2gray(img)
# fig, axs = plt.subplots(1,2,figsize=(10,6))
# axs[0].imshow(img, cmap='Greys_r')
# m = np.ma.masked_where(best_mask == 0, best_mask)
# axs[0].imshow(m, cmap = matplotlib.colors.ListedColormap(['white', 'crimson']), \
#               vmin=0, vmax=1, alpha=0.2, zorder=1)
# axs[0].set_title('Image and overlay of the best mask')
# axs[1].imshow(best_mask, cmap='Greys_r', vmin=0, vmax=1)
# axs[1].set_title('Best mask')
# fig.tight_layout()
# plt.show()
#
# # %% Check bright images
# img_mean = []
# bright = []
# for i in range(fn.shape[0]):
#     m = np.mean(skimage.io.imread(IN_DATA_PATH + fn.iloc[i]))
#     if m > 125: bright.append(fn.iloc[i])
#     img_mean.append(m)
#
# fig, axs = plt.subplots(20,10, figsize=(10,20))
# for f, ax in zip(bright, axs.reshape(-1)):
#     ax.set_axis_off()
#     ax.imshow(skimage.io.imread(IN_DATA_PATH + f), cmap='Greys_r')
# fig.tight_layout()
# plt.show()
#
# fig, ax = plt.subplots(1,1,figsize=(9,7))
# ax.hist(img_mean, color='gray', bins=150)
# plt.show()
#
#
#
#
#
#
#
#
#
# # %% ##########
# import time
# IN_DATA_PATH = '../../../data/RAW/'
# OUT_DATA_PATH = '../../../data/PROCESSED/'
# DATAINFO_PATH = '../../../data/'
#
# df = pd.read_csv(DATAINFO_PATH + 'data_info.csv')
# fn = df.loc[df.body_part == 'HAND', 'filename']
# img = skimage.io.imread(IN_DATA_PATH + fn.iloc[589])
#
# plt.imshow(img, cmap='Greys_r')
# #%%
# t = []
# for _ in range(20):
#     t0 = time.time()
#     mask = find_best_mask(img, tol_factor=0.1, search_res=0.1, search_range=(0.2,1))
#     t1 = time.time()
#     t.append(t1-t0)
# t = np.array(t)
# print(f'{t.mean()} +/- {t.std()}')
#
# # %%
# t0 = time.time()
# mask = find_best_mask(img, tol_factor=0.1, search_res=0.1, search_range=(0.2,1))
# skimage.io.imsave('test_m.png', skimage.img_as_ubyte(mask), check_contrast=False)
# t1 = time.time()
# print(t1-t0)
# plt.imshow(mask, cmap='Greys_r')
