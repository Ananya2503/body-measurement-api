# import necessary libraries
import numpy as np
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths
from pathlib import Path
from keras.preprocessing.image import save_img
import cv2 as cv
from measure import *

num = 5
FRONT = 'front'
SIDE = 'side'
HEIGHT = 164
# 43 81 70 88
# 44 86 80 98

# setup input and output path
def setupPath():
    output_path = Path('./output')
    output_path.mkdir(parents=True, exist_ok=True)
    print("setup path finish")
    return output_path

# get prediction result
def predict():
    front_img = cv.imread(f'./image/{FRONT}-{num}.jpg')
    side_img = cv.imread(f'./image/{SIDE}-{num}.jpg')
    width =  int(front_img.shape[0] * 0.695)
    # height = width
    front_img = cv.resize(front_img, (width, width))
    side_img = cv.resize(side_img, (width,width))
    front_result = bodypix.predict_single(front_img)
    side_result = bodypix.predict_single(side_img)

    # mask
    front_simple_mask = getSimpleMask(front_result, FRONT)
    side_simple_mask = getSimpleMask(side_result, SIDE)
    getColorMask(front_result, front_simple_mask, FRONT)
    getColorMask(front_result, side_simple_mask, SIDE)

    # get user height
    result_front_img = cv.imread(f'{output_path}/front-simple-mask.jpg')
    result_side_img = cv.imread(f'{output_path}/side-simple-mask.jpg')
    user_height_pixel_front, max_coor_front = getHeightInPixel(result_front_img, width, width)
    user_height_pixel_side, max_coor_side = getHeightInPixel(result_side_img, width, width)
    # print(user_height_pixel_front, user_height_pixel_side)

    # get body part position
    shoulder_front_position, chest_front_position, waist_front_position, hip_front_position = getBodyProportion(user_height_pixel_front, max_coor_front)
    shoulder_side_position, chest_side_position, waist_side_position, hip_side_position = getBodyProportion(user_height_pixel_side, max_coor_side)

    # measure
    shoulder, chest, waist, hip = measure([shoulder_front_position, shoulder_side_position],
            [chest_front_position, chest_side_position],
            [waist_front_position, waist_side_position],
            [hip_front_position, hip_side_position],
            width, HEIGHT, user_height_pixel_front, user_height_pixel_side)
    return shoulder, chest, waist, hip

# simple mask
def getSimpleMask(result, pose):
    mask = result.get_mask(threshold=0.5)
    save_img(f'{output_path}/{pose}-simple-mask.jpg', mask)
    print("simple mask finished")
    return mask

# color mask
def getColorMask(result, mask, pose):
    color_mask = result.get_colored_part_mask(mask)
    save_img(f'{output_path}/{pose}-color-mask.jpg', color_mask)
    print("color mask finished")

if __name__ == '__main__':
    output_path = setupPath()
    bodypix = load_model(download_model(BodyPixModelPaths.RESNET50_FLOAT_STRIDE_16), output_stride=16)
    shoulder, chest, waist, hip = predict()
    print("shoulder:", shoulder)
    print("chest:", chest)
    print("waist:", waist)
    print("hip:", hip)