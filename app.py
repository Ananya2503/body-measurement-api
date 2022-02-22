# import necessary libraries
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths
from pathlib import Path
from keras.preprocessing.image import save_img
import cv2 as cv
from measure import *

num = 2
FRONT = 'front'
SIDE = 'side'

# setup input and output path
def setupPath():
    output_path = Path('./output')
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

# load model
def loadModel() :
    print("load model ...")
    bodypix = load_model(download_model(BodyPixModelPaths.RESNET50_FLOAT_STRIDE_16), output_stride=16)
    return bodypix

# get prediction result
def predict():
    # scale = 50
    front_img = cv.imread(f'./image/{FRONT}-{num}.jpg')
    side_img = cv.imread(f'./image/{SIDE}-{num}.jpg')
    # height = int(front_img.shape[0] * scale / 100)
    # width = int(front_img.shape[1] * scale / 100)
    height = 800
    width = 800
    front_img = cv.resize(front_img, (width, height))
    side_img = cv.resize(side_img, (width,height))
    front_result = bodypix.predict_single(front_img)
    side_result = bodypix.predict_single(side_img)
    print("Prediction finish")

    # mask
    front_simple_mask = getSimpleMask(front_result, FRONT)
    side_simple_mask = getSimpleMask(side_result, SIDE)
    # getColorMask(front_result, front_simple_mask, FRONT)
    # getColorMask(front_result, side_simple_mask, SIDE)

    # get user height
    result_front_img = cv.imread(f'{output_path}/front-simple-mask.jpg')
    result_side_img = cv.imread(f'{output_path}/side-simple-mask.jpg')
    user_height_pixel_front, max_coor_front, min_coor_front = getHeightInPixel(result_front_img, width, height)
    user_height_pixel_side, max_coor_side, min_coor_side = getHeightInPixel(result_side_img, width, height)

    # get body part position
    shoulder_front_position, chest_front_position, waist_front_position, hip_front_position = getBodyProportion(result_front_img, width, height, user_height_pixel_front, max_coor_front, min_coor_front)
    shoulder_side_position, chest_side_position, waist_side_position, hip_side_position = getBodyProportion(result_side_img, width, height, user_height_pixel_side, max_coor_side, min_coor_side)
    print(shoulder_front_position, chest_front_position, waist_front_position, hip_front_position)
    print(shoulder_side_position, chest_side_position, waist_side_position, hip_side_position)
    return user_height_pixel_front

# simple mask
def getSimpleMask(result, pose):
    mask = result.get_mask(threshold=0.5)
    save_img(f'{output_path}/{pose}-simple-mask.jpg', mask)
    print("Simple mask finish")
    return mask

# color mask
def getColorMask(result, mask, pose):
    color_mask = result.get_colored_part_mask(mask)
    save_img(f'{output_path}/{pose}-color-mask.jpg', color_mask)
    print('color mask finish')

if __name__ == '__main__':
    output_path = setupPath()
    bodypix = loadModel()
    user_height_pixel = predict()
    print(user_height_pixel)