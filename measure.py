from PIL import Image
from math import pi, sqrt

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
    return min_height[1] - max_height[1], max_height

# get body proportion
def getBodyProportion(user_height_pixel, max_coor):
    section_height = int(user_height_pixel / 8)

    # shoulder
    shoulder = int(max_coor[1] + section_height + (section_height / 2))

    # chest
    chest = max_coor[1] + (section_height * 2)

    # waist
    waist = max_coor[1] + (section_height * 3)

    # hip
    hip = int(max_coor[1] + (3 * section_height) + (section_height / 2))

    return shoulder, chest, waist, hip

def measure(shoulder_point, chest_point, waist_point, hip_point, width):
    front_img = Image.open(f'output/front-color-mask.jpg')
    side_img = Image.open(f'output/side-color-mask.jpg')

    # shoulder (only front)
    shoulder = getDistant(front_img, shoulder_point[0], width)

    # chest
    chest_front = getDistant(front_img, chest_point[0], width)
    chest_side = getDistant(side_img, chest_point[1], width)
    chest = getPerimeter(chest_front, chest_side)

    # waist
    waist_front = getDistant(front_img, waist_point[0], width)
    waist_side = getDistant(side_img, waist_point[1], width)
    waist = getPerimeter(waist_front, waist_side)

    # hip
    hip_front = getDistant(front_img, hip_point[0], width)
    hip_side = getDistant(side_img, hip_point[1], width)
    hip = getPerimeter(hip_front, hip_side)
   
    return shoulder, chest, waist, hip


def getDistant(image, point, width):
    R_BASE = range(128, 204)
    G_BASE = range(203, 256)
    B_BASE = range(64, 152)

    border_point = [0, 0] # [left, right]
    count = 0
    px = image.load()

    for i in range(width):
        (r, g, b) = px[i, point]
        if (r in R_BASE) and (g in G_BASE) and (b in B_BASE):
            if border_point[0] == 0 or (border_point[0] != 0 and count > 50):
                border_point[0] = i
            elif border_point[0] != 0:
                border_point[1] = i
            count = 0
        else:
            count += 1
    distance = sqrt(pow((border_point[1] - border_point[0]), 2))
    return int(distance)

def getPerimeter(front_point, side_point):
    front_distance = int(front_point / 2)
    side_distance = int(side_point / 2)

    perimeter = 2 * pi * sqrt((pow(front_distance, 2) + pow(side_distance, 2)) / 2)
    return int(perimeter)