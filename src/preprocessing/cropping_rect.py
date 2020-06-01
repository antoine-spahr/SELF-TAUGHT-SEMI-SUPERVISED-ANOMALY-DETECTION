import numpy as np
import cv2 as cv
import imutils
import os
import sys
from glob import glob
#import matplotlib.pyplot as plt
import skimage
#sys.path.append('../')
#from ..utils.utils import print_progessbar, print_param_summary

IN_DATA_PATH = '../../../data/RAW/'
OUT_DATA_PATH = '../../../data/PROCESSED/'

def angle_cos(p0, p1, p2):
    """

    """
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

def find_squares(img, min_area=50000, max_skew=0.45):
    """
    A method to find inner square images on bigger images
    :param min_area: specifies minimal square area in pixels
    :param max_skew: specifies maximum skewness of squares
    :param img: numpy array representation of an image
    :return: list of found squares
    """
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            bin = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(bin)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) >= 4 and cv.contourArea(cnt) >= min_area \
                        and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4],
                                                cnt[(i + 2) % 4])
                                      for i in range(4)])
                    if max_cos < max_skew:
                        squares.append(cnt)
    return squares


def crop_squares(squares, img):
    """

    """
    rect = cv.minAreaRect(squares[0])
    box = cv.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]],
                       dtype="float32")

    # the perspective transformation matrix
    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(img, M, (width, height))

    if width > height:
        warped = imutils.rotate_bound(warped, 270)

    return warped

# def main():
#     """
#
#     """
#     print('-'*100 + '\n' + 'PREPROCESSING'.center(100) + '\n' + '-'*100)
#     summary = {}
#     # iterate body part
#     for bpdir in glob(IN_DATA_PATH + '/*/'):
#         bpname = bpdir.split("/")[-2][3:].title()
#         print(f'|--- Processing {bpname} X-rays')
#         XR_count = 0
#         patient_count = 0
#         missing_patients = 0
#         uncropped_img = 0
#         # iterate patient
#         n_dir = len(glob(bpdir + '/*/'))
#         for i, pdir in enumerate(glob(bpdir + '/*/')):
#             has_img = False
#             patient_count += 1
#             # iterate over studies
#             for sdir in glob(pdir + '/*/'):
#                 # create the write dir if it does not exist
#                 write_dir = OUT_DATA_PATH + '/'.join(sdir.split('/')[-4:-1])
#                 if not os.path.isdir(write_dir):
#                     os.makedirs(write_dir)
#                 # iterate over images
#                 for fn in glob(sdir + '/*'):
#                     XR_count += 1
#                     has_img = True
#                     img = skimage.io.imread(fn)
#                     # get squares
#                     squares = find_squares(img, min_area=50000)
#
#                     if squares:
#                         warped = crop_squares(squares, img)
#                         skimage.io.imsave(write_dir + "/" + os.path.basename(fn), warped)
#                     else:
#                         skimage.io.imsave(write_dir + "/" + os.path.basename(fn), img)
#                         uncropped_img += 1
#
#             print_progessbar(i, n_dir, Name='|------ Patients', Size=50)
#             if not has_img: missing_patients += 1
#         summary[bpname] = {'X-ray':XR_count, 'patient':patient_count, \
#                            'missing patient':missing_patients, 'uncropped images':uncropped_img}
#         print('')
#     print_param_summary(**summary)
#
# if __name__ == '__main__':
#     print(__doc__)
#     main()
