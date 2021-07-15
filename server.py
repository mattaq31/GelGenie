from aiohttp import web
import socketio
import os
from gui.lane_finder import LaneFinder
from gel_tools.band_finder import find_bands, load_image
import cv2
from matplotlib import image
from matplotlib import pyplot as plt
import skimage
from PIL import Image
import json
import numpy as np

import base64
from io import BytesIO

import jsonpickle




# allows one to run server from any base filepath
os.chdir(os.path.abspath(os.path.join(__file__, os.path.pardir)))


# Create an async aiohttp server
sio = socketio.AsyncServer(cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

@sio.on("imageToRead")
async def imageToRead(sid, file):
    print("Finding bands in ", file)
    image = load_image(file)
    result = find_bands(image)
    saved_file = plt.imsave("results/result1.jpg", result[0])
    # im = Image.fromarray(result)
    # im.save("results/result1.jpg")
    # skimage.io.imsave("results/result1.jpg", result)
    # cv2.imwrite("results/result1.jpg", result)

    pil_img = Image.fromarray(np.uint8(result[0]*255))
    # pil_img = pil_img.convert('RGB')
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("ascii")


    # print(new_image_string)
    print(result[1])
    band_centroids = []
    for region_object in result[1]:
        if region_object.area < 50:
            continue
        else:
            band_centroids.append(region_object.centroid)
    print(band_centroids)
    encoded_centroids = json.dumps(band_centroids)
    band_props = jsonpickle.encode(result[1])
    await sio.emit('viewResult', {'file': new_image_string, 'props': band_props, 'centroids': encoded_centroids})

# Define aiohttp endpoints
# This will deliver the main.html file to the client once connected.
async def index(request):
    print("request")
    with open('gui/main.html') as f:
        return web.Response(text=f.read(), content_type='html')

# Bind our aiohttp endpoint to our app router
app.router.add_get('/', index)

# Add the static css and js files to the app router
app.router.add_static('/', 'gui/static')

# We kick off our server
if __name__ == '__main__':
    web.run_app(app)

async def button_find_bands(request):
    return web.Response()



