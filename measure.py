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
    user_height = min_height[1] - max_height[1]
    return user_height, max_height

# get body proportion
def getBodyProportion(user_height_pixel):
    section_height = int(user_height_pixel / 8)

    # shoulder
    shoulder = int(section_height + (section_height / 2))

    # chest
    chest = section_height * 2

    # waist
    waist = section_height * 3

    # hip
    hip = int((3 * section_height) + (section_height / 2))

    return shoulder, chest, waist, hip

def measure(shoulder_point, chest_point, waist_point, hip_point, user_height, user_height_pixel_front, user_height_pixel_side):
    front_img = Image.open(f'output/front-color-mask.jpg')
    side_img = Image.open(f'output/side-color-mask.jpg')

    # ratio
    ratio_front = user_height / user_height_pixel_front
    ratio_side = user_height / user_height_pixel_side

    # shoulder (only front)
    shoulder = int(getDistant(front_img, shoulder_point[0]) * ratio_front * 1.3)

    # chest
    chest_front = getDistant(front_img, chest_point[0]) * ratio_front
    chest_side = getDistant(side_img, chest_point[1]) * ratio_side
    chest = getPerimeter(chest_front, chest_side)

    # waist
    waist_front = getDistant(front_img, waist_point[0]) * ratio_front
    waist_side = getDistant(side_img, waist_point[1]) * ratio_side
    waist = getPerimeter(waist_front, waist_side)

    # hip
    hip_front = getDistant(front_img, hip_point[0]) * ratio_front
    hip_side= getDistant(side_img, hip_point[1]) * ratio_side
    hip = getPerimeter(hip_front, hip_side)
   
    return shoulder, chest, waist, hip


def getDistant(image, point):
    R_BASE = range(128, 204)
    G_BASE = range(203, 256)
    B_BASE = range(64, 152)

    border_point = [0, 0] # [left, right]
    count = 0
    width = image.size[0]
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
    distance = sqrt((border_point[1] - border_point[0])**2)
    return distance

def getPerimeter(front_point, side_point):
    a = front_point / 2
    b = side_point / 2
    # h = ((a - b)**2) / ((a + b)**2)
    # perimeter = pi * (a + b)
    # perimeter = pi * sqrt( 2 * ((a**2) + (b**2)))
    perimeter = pi * ((3 / 2) * (a + b) - sqrt(a * b))
    # perimeter = pi * (3 * (a + b) - sqrt((3 * a + b) * (a + 3 * b)))
    # perimeter = pi * (a + b) * (1 + ((3 * h) / (10 + sqrt(4 - (3 * h)))))
    # perimeter = 2 * pi * sqrt(((a**2) + (b**2)) / 2)
    # perimeter = pi * (a + b) * (3 * (((a - b)**2) / (((a + b)**2) * (sqrt(-3 * h + 4) + 10))) + 1)
    return int(perimeter)