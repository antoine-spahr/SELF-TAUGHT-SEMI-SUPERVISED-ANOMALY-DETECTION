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
