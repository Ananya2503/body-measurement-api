# import necessary libraries
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths
from pathlib import Path
from keras.preprocessing.image import save_img
import cv2 as cv

num = 2
pose = 'front'

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
    scale = 50
    img = cv.imread(f'./image/{pose}-{num}.jpg')
    height = int(img.shape[0] * scale / 100)
    width = int(img.shape[1] * scale / 100)
    img = cv.resize(img, (width, height))
    result = bodypix.predict_single(img)
    print("Prediction finish")

    # measure
    simple_mask = getSimpleMask(result)
    result_img = cv.imread(f'{output_path}/{pose}-simple-mask-{num}.jpg')
    user_height_pixel = getHeightInPixel(result_img, width, height)
    return user_height_pixel

# simple mask
def getSimpleMask(result):
    mask = result.get_mask(threshold=0.5)
    save_img(f'{output_path}/{pose}-simple-mask-{num}.jpg', mask)
    print("Simple mask finish")
    return mask

# color mask
def getColorMask(result, mask):
    color_mask = result.get_colored_part_mask(mask)
    save_img(f'{output_path}/{pose}-color-mask-{num}.jpg', color_mask)
    print('color mask finish')

# find user height (pixel)
def getHeightInPixel(image, width, height):
    max_height = [0, 0] # [x, y]
    min_height = [0, 0]
    for i in range(height): # y
        for j in range(width): # x
            (b, g ,r) = image[i, j]
            if (b == 255 and g == 255 and r == 255):
                if max_height[1] > i or max_height[1] == 0:
                    max_height = [j, i]
                if min_height[1] < i:
                    min_height = [j, i]
    print("max height: ", max_height)
    print("min height: ", min_height)
    circle_img = image
    # cv.circle(circle_img, (max_height[0], max_height[1]), 20, (255, 0, 0), -1)
    # cv.circle(circle_img, (max_height[0], min_height[1]), 20, (255, 0, 0), -1)
    # cv.line(circle_img, (max_height[0], max_height[1]), (max_height[0], min_height[1]), (255, 0, 0), 15)
    # save_img(f'{output_path}/{pose}-test-height-{num}.jpg', circle_img)
    return min_height[1] - max_height[1]

# measure

if __name__ == '__main__':
    output_path = setupPath()
    bodypix = loadModel()
    user_height_pixel = predict()