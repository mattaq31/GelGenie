from aiohttp import web
import socketio
import os
from gui.lane_finder import LaneFinder
from gel_tools.band_finder import find_bands, load_image, load_image_b64
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
async def imageToRead(sid, source_b64):
    # print("Finding bands in ", file)
    # print(source_b64)
    (image, non_np_image, width, height) = load_image_b64(source_b64)

    buff = BytesIO()
    non_np_image.save(buff, format="PNG")
    source_image_jpg = base64.b64encode(buff.getvalue()).decode("ascii")
    await sio.emit('sourceInJpg', source_image_jpg)

    # Find the bands using watershed segmentation
    result = find_bands(image)

    # Convert numpy image to PIL image
    pil_img = Image.fromarray(np.uint8(result[0]*255))
    # Convert inverted numpy image to PIL image
    pil_img_inverted = Image.fromarray(np.uint8(result[2]*255))

    buff2 = BytesIO()
    pil_img.save(buff2, format="JPEG")
    new_image_string = base64.b64encode(buff2.getvalue()).decode("ascii")

    buff3 = BytesIO()
    pil_img_inverted.save(buff3, format="JPEG")
    inverted_image_string = base64.b64encode(buff3.getvalue()).decode("ascii")


    # print(new_image_string)
    print(result[1])
    band_centroids = []
    band_areas = []
    band_weighted_areas = []
    bboxs = []
    for region_object in result[1]:
        if region_object.area < 50:
            continue
        else:
            band_centroids.append(region_object.centroid)
            band_areas.append(region_object.area.item())

            # Calculate and append weighted area
            weighted_area = round(region_object.mean_intensity.item() * region_object.area.item() / (255*255))
            band_weighted_areas.append(weighted_area)

            bboxs.append(region_object.bbox)

    print(band_centroids)
    encoded_centroids = json.dumps(band_centroids)
    encoded_areas = json.dumps(band_areas)
    encoded_w_areas = json.dumps(band_weighted_areas)
    band_props = jsonpickle.encode(result[1])
    encoded_bboxs = json.dumps(bboxs)
    await sio.emit('viewResult', {'file': new_image_string, 'inverted_file': inverted_image_string,
                                  'im_width': width, 'im_height': height, 'props': band_props,
                                  'centroids': encoded_centroids, 'bboxs': encoded_bboxs,
                                  'areas': encoded_areas, 'w_areas': encoded_w_areas})

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



