import cv2 as cv
from keras.preprocessing.image import save_img

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
    # circle_img = image
    # cv.circle(circle_img, (max_height[0], max_height[1]), 20, (255, 0, 0), -1)
    # cv.circle(circle_img, (max_height[0], min_height[1]), 20, (255, 0, 0), -1)
    # cv.line(circle_img, (max_height[0], max_height[1]), (max_height[0], min_height[1]), (255, 0, 0), 15)
    # save_img(f'output/test-height.jpg', circle_img)
    return min_height[1] - max_height[1], max_height, min_height

# get head height
def getHeadHeight(image, width, height):
    max_height = [0, 0] # [x, y]
    min_height = [0, 0]
    for i in range(height): # y
        for j in range(width): # x
            (b, g ,r) = image[i, j]
            if (b == 170 and g == 65 and r == 110) or (b == 179 and g == 61 and r == 144): # right or left
                if max_height[1] > i or max_height[1] == 0:
                    max_height = [j, i]
                if min_height[1] < i:
                    min_height = [j, i]
    circle_img = image
    cv.circle(circle_img, (max_height[0], max_height[1]), 20, (255, 0, 0), -1)
    cv.circle(circle_img, (max_height[0], min_height[1]), 20, (255, 0, 0), -1)
    cv.line(circle_img, (max_height[0], max_height[1]), (max_height[0], min_height[1]), (255, 0, 0), 15)
    save_img(f'output/test-head-height.jpg', circle_img)
    return min_height[1] - max_height[1]

# get body proportion
def getBodyProportion(image, width, height, user_height_pixel, max_coor, min_coor):
    section_height = int(user_height_pixel / 8)
    circle_img = image
    # shoulder
    cv.line(circle_img, (0, int(max_coor[1] + section_height + (section_height / 2))), (width - 1, int(max_coor[1] + section_height + (section_height / 2))), (255, 0, 0), 15)
    # chest
    cv.line(circle_img, (0, max_coor[1] + (section_height * 2)), (width - 1, max_coor[1] + (section_height * 2)), (255, 0, 0), 15)
    # waist
    cv.line(circle_img, (0, max_coor[1] + (section_height * 3)), (width - 1, max_coor[1] + (section_height * 3)), (255, 0, 0), 15)
    # hip
    cv.line(circle_img, (0, int(max_coor[1] + (3 * section_height) + (section_height / 2))), (width - 1, int(max_coor[1] + (3 * section_height) + (section_height / 2))), (255, 0, 0), 15)
    save_img(f'output/test-body-section.jpg', circle_img)
