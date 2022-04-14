# import necessary libraries
from flask import Flask, request, jsonify
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths
from pathlib import Path
from keras.preprocessing.image import save_img
import cv2 as cv
import numpy as np
from measure import get_body_proportion, get_height_in_pixel, measure

app = Flask(__name__)

FRONT = 'front'
SIDE = 'side'
SCALE = 0.28
# 43 81 70 88
# 44 86 80 98

# setup output path
def setup_path():
    output_path = Path('./output')
    output_path.mkdir(parents=True, exist_ok=True)

# remove output directory
def remove_dir():
    output_file = Path('./output/').glob('*.jpg')
    output_path = Path('./output')
    
    for f in output_file:
        try:
            f.unlink()
        except OSError as e:
            print('Error: %s : %s' % (f, e.strerror))
    try:
        output_path.rmdir()
    except OSError as e:
        print('Error: %s : %s', (output_path, e.strerror))

# get prediction result
@app.route('/measure', methods=['POST'])
def predict():
    if request.method == 'POST':
        # get data
        if request.form.get('height'):
            user_height = int(request.form['height'])
        else:
            return "กรุณาระบุส่วนสูง"
        if request.files.get('front'):
            front = request.files['front'].read()
        else:
            return "กรุณาระบุภาพด้านหน้า"
        if request.files.get('side'):
            side = request.files['side'].read()
        else:
            return "กรุณาระบุภาพด้านข้าง"

        # setup
        setup_path()
        bodypix = load_model(download_model(BodyPixModelPaths.RESNET50_FLOAT_STRIDE_16), output_stride=16)

        # image processing
        front_img = img_process(front)
        side_img = img_process(side)
        width = front_img.shape[1]
        height = front_img.shape[0]
        
        # predict
        front_result = bodypix.predict_single(front_img)
        side_result = bodypix.predict_single(side_img)

        # mask
        front_simple_mask = get_simple_mask(front_result, FRONT)
        side_simple_mask = get_simple_mask(side_result, SIDE)
        get_color_mask(front_result, front_simple_mask, FRONT)
        get_color_mask(front_result, side_simple_mask, SIDE)

        # get user height
        result_front_img = cv.imread(f'output/front-simple-mask.jpg')
        result_side_img = cv.imread(f'output/side-simple-mask.jpg')
        user_height_pixel_front, max_coor_front = get_height_in_pixel(result_front_img, width, height)
        user_height_pixel_side, max_coor_side = get_height_in_pixel(result_side_img, width, height)

        # crop image
        result_front_color = cv.cvtColor(cv.imread(f'output/front-color-mask.jpg'), cv.COLOR_BGR2RGB)
        result_front_color = crop_image(result_front_color, max_coor_front, user_height_pixel_front, FRONT)
        result_side_color = cv.cvtColor(cv.imread(f'output/side-color-mask.jpg'), cv.COLOR_BGR2RGB)
        result_side_color = crop_image(result_side_color, max_coor_side, user_height_pixel_side, SIDE)

        # get body part position
        shoulder_front_position, chest_front_position, waist_front_position, hip_front_position = get_body_proportion(user_height_pixel_front)
        shoulder_side_position, chest_side_position, waist_side_position, hip_side_position = get_body_proportion(user_height_pixel_side)

        # measure
        shoulder, chest, waist, hip = measure([shoulder_front_position, shoulder_side_position],
            [chest_front_position, chest_side_position],
            [waist_front_position, waist_side_position],
            [hip_front_position, hip_side_position],
            user_height, user_height_pixel_front, user_height_pixel_side)
        
        # clear output directory
        remove_dir()

        # check data before return
        if (shoulder == 0) or (chest == 0) or (waist == 0) or (hip == 0):
            return "กรุณาถ่ายภาพใหม่อีกครั้ง"
        else:
            return jsonify({
                'shoulder': shoulder,
                'chest': chest,
                'waist': waist,
                'hip': hip
            })
    else:
        return "Invalid method"

# simple mask
def get_simple_mask(result, pose):
    mask = result.get_mask(threshold=0.5)
    save_img(f'output/{pose}-simple-mask.jpg', mask)
    return mask

# color mask
def get_color_mask(result, mask, pose):
    color_mask = result.get_colored_part_mask(mask)
    save_img(f'output/{pose}-color-mask.jpg', color_mask)

# crop image
def crop_image(image, max_coor, user_height, pose):
    crop_image = image[max_coor[1]:max_coor[1] + user_height + 1, max_coor[0] - int(user_height / 2):max_coor[0] + int(user_height / 2) + 1] # [height, width]
    save_img(f'output/{pose}-color-mask.jpg', crop_image)
    return crop_image

# image processing
def img_process(image):
    img = np.frombuffer(image, np.uint8)
    img = cv.imdecode(img, cv.IMREAD_UNCHANGED)
    img = cv.resize(img, (int(img.shape[1] * SCALE), int(img.shape[0] * SCALE)))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    return img

@app.route('/', methods=['GET'])
def home():
    return "Body measurement API"

if __name__ == '__main__':
    app.run()