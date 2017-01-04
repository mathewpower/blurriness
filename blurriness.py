#!/usr/bin/env python

import cv2
import os
import logging

logging.basicConfig()

logger = logging.getLogger('blurriness')
logger.setLevel(logging.DEBUG)


def is_blurry(image, threshold):
    '''
    Return True if the image blurriness exceeds the threshold.

    The lower the variance, the lower the edge definition.
    Lower edge definition means more blurry.
    '''
    matrix = cv2.Laplacian(image, cv2.CV_64F)
    variance = matrix.var(dtype=float)

    logger.debug('image has a variance of {}'.format(variance))

    return variance < threshold


image_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref')


# Blurry image
image_path = os.path.join(image_base_path, 'blur.jpg')
image = cv2.imread(image_path)

assert is_blurry(image, threshold=1000)
logger.info('{} is blurry'.format(image_path)) 


# Not blurry image
image_path = os.path.join(image_base_path, 'no-blur.jpg')
image = cv2.imread(image_path)

assert not is_blurry(image, threshold=1000)
logger.info('{} is not blurry'.format(image_path)) 

