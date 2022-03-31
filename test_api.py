import cv2 as cv
from flask import json
from app import app

SCALE = 0.28

def test_api():
    front = cv.imread(f'image/front-5.jpg')
    side = cv.imread(f'image/side-5.jpg')

    front = cv.resize(front, (int(front.shape[1] * SCALE), int(front.shape[0] * SCALE)))
    side = cv.resize(side, (int(side.shape[1] * SCALE), int(side.shape[0] * SCALE)))

    front = front.tolist()
    side = side.tolist()

    # sent HTTP request
    data = json.dumps({
        'height': '164',
        'front': front,
        'side': side
    }).encode('utf-8')


    res = app.test_client().post(
        '/measure',
        data=data,
        content_type='application/json'
    )

    print(res.data)
