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
from collections import defaultdict
import pandas as pd


# allows one to run server from any base filepath
os.chdir(os.path.abspath(os.path.join(__file__, os.path.pardir)))


# Create an async aiohttp server
sio = socketio.AsyncServer(cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

full_results = None;
original_image = None;

@sio.on("imageToRead")
async def imageToRead(sid, source_b64):
    """Takes a user-selected image from JS, finds bands and properties, and returns these to JS"""

    # Load image from b64 received from JS, giving width and height
    (image, non_np_image, width, height) = load_image_b64(source_b64)


    # Encode original image to b64 to display while bands are found
    # Essentially to convert .tif from JS into .jpg which browsers can show
    buff = BytesIO()
    non_np_image.save(buff, format="PNG")
    source_image_jpg = base64.b64encode(buff.getvalue()).decode("ascii")
    await sio.emit('sourceInJpg', source_image_jpg)


@sio.on("findBands")
async def findBands(sid, source_b64):
    # Load image from b64 received from JS, giving width and height
    (image, non_np_image, width, height) = load_image_b64(source_b64)
    global original_image
    original_image = image

    # Find the bands using watershed segmentation
    result = find_bands(image)

    # Convert numpy image to PIL image
    pil_img = Image.fromarray(np.uint8(result[0]*255))
    # Convert inverted numpy image to PIL image
    pil_img_inverted = Image.fromarray(np.uint8(result[2]*255))

    # Attempts at converting to 16 bit images follow
    """
    _, imagebytes = cv2.imencode('.jpg', np.uint16(result[0]*65535))
    new_image_string = base64.b64encode(imagebytes)
    print("This is the b64 encoded string")
    print(new_image_string)
    _, imagebytes2 = cv2.imencode('.jpg', np.uint16(result[2] * 65535))
    inverted_image_string = base64.b64encode(imagebytes2)
    """
    _, imagebytes = cv2.imencode('.png', np.uint16(result[0] * 65535))
    new_image_string = base64.b64encode(imagebytes).decode('utf-8')
    # print("This is the b64 encoded string")
    # print(new_image_string)
    _, imagebytes2 = cv2.imencode('.png', np.uint16(result[2] * 65535))
    inverted_image_string = base64.b64encode(imagebytes2).decode('utf-8')

    """
    retval, buffer10 = cv2.imencode('.jpg', np.uint16(result[0]*65535))
    im_bytes10 = buffer10.tobytes()
    str10 = cv2.imencode('.jpg', np.uint16(result[0]*65535))[1].tostring()
    new_image_string = base64.b64encode(str10)
    retval2, buffer11 = cv2.imencode('.jpg', np.uint16(result[2] * 65535))
    im_bytes11 = buffer11.tobytes()
    inverted_image_string = base64.b64encode(im_bytes11)
    """

    """
    bytesio = BytesIO()
    np.savetxt(bytesio, np.uint16(result[0]*65535))  # Only supports 1D or 2D arrays, numpy arrays are converted into byte streams
    content = bytesio.getvalue()  # Get string representation
    print(content)
    new_image_string = base64.b64encode(content)

    bytesio2 = BytesIO()
    np.savetxt(bytesio2, np.uint16(result[2] * 65535))  # Only supports 1D or 2D arrays, numpy arrays are converted into byte streams
    content2 = bytesio2.getvalue()  # Get string representation
    print(content2)
    inverted_image_string = base64.b64encode(content2)
    """

    """
    # Encode image to b64
    buff2 = BytesIO()
    pil_img.save(buff2, format="JPEG")
    new_image_string = base64.b64encode(buff2.getvalue()).decode("ascii")
    # Encode inverted image to b64
    buff3 = BytesIO()
    pil_img_inverted.save(buff3, format="JPEG")
    inverted_image_string = base64.b64encode(buff3.getvalue()).decode("ascii")
    print(result[1])
    """

    # Create lists of props to pass to JS
    band_centroids = []
    band_areas = []
    band_weighted_areas = []
    bboxs = []
    band_labels = []
    band_indices = []

    # Create band dictionary
    band_keys = ["id", "label", "center_x", "center_y", "area", "w_area", "bbox"]
    band_dict = defaultdict(list)

    band_no = 1
    # Filter bands with area less than 50, and find props of those larger than 50
    for region_object in result[1]:
        if region_object.area < 50:
            continue
        else:
            # Add relevant band props to lists (to be deprecated)
            band_centroids.append(region_object.centroid)
            band_areas.append(region_object.area.item())
            # Calculate and append weighted area
            weighted_area = round(region_object.mean_intensity.item() * region_object.area.item() / (255*255))
            band_weighted_areas.append(weighted_area)
            bboxs.append(region_object.bbox)
            band_indices.append(band_no)
            band_labels.append(band_no)

            # Add relevant props to dictionary
            band_dict["id"].append(band_no) # cannot be changed by user
            band_dict["label"].append(band_no) # default label is band number, but can be changed
            band_dict["center_x"].append(region_object.centroid[1])
            band_dict["center_y"].append(region_object.centroid[0])
            band_dict["area"].append(region_object.area.item())
            band_dict["w_area"].append(weighted_area)
            band_dict["c_area"].append(weighted_area)
            band_dict["bbox"].append(region_object.bbox)
            band_no += 1

    print(band_dict)
    indexes = ["id"]  # place your main indices here (for example, it could be the band number)

    # converts dictionary into a dataframe, and sets your selected indices as your row references
    global full_results
    full_results = pd.DataFrame.from_dict(band_dict).set_index(indexes)

    print(full_results)
    # Export pandas table to csv
    # full_results.to_csv("test_metrics.csv")

    # Encode selected band props to json
    encoded_centroids = json.dumps(band_centroids)
    encoded_areas = json.dumps(band_areas)
    encoded_w_areas = json.dumps(band_weighted_areas)
    band_props = jsonpickle.encode(result[1])
    encoded_bboxs = json.dumps(bboxs)
    encoded_indices = json.dumps(band_indices)
    encoded_labels = json.dumps(band_labels)
    # Send image and band props to JS
    await sio.emit('viewResult', {'file': new_image_string, 'inverted_file': inverted_image_string,
                                  'im_width': width, 'im_height': height, 'props': band_props,
                                  'centroids': encoded_centroids, 'bboxs': encoded_bboxs,
                                  'areas': encoded_areas, 'w_areas': encoded_w_areas,
                                  'indices': encoded_indices, 'labels': encoded_labels})

@sio.on("updateBandLabel")
async def updateBandLabel(sid, band_id, updated_label):
    full_results.loc[band_id, "label"] = updated_label
    print("The band id is: ", band_id, " and the updated label is ", updated_label)
    new_labels = json.dumps(full_results["label"].tolist())
    await sio.emit("labelUpdated", new_labels)

@sio.on("exportToCsv")
async def exportToCsv(sid):
    # Export pandas table to csv
    full_results.to_csv("test_metrics.csv")

@sio.on("laneProfile")
async def laneProfile(sid, x_pos, y_end):
    img = original_image
    profile = skimage.measure.profile_line(img, [0, x_pos], [y_end, x_pos], linewidth=7, reduce_func=np.mean)
    plt.figure(figsize=(4, 11), facecolor='black')
    plt.plot(profile, range(len(profile)))
    plt.gca().invert_yaxis()
    plt.axis("off")
    my_stringIObytes = BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('utf-8')
    print("Found lane profile")
    await sio.emit('foundLaneProfile', my_base64_jpgData)


@sio.on("calibrateArea")
async def calibrateArea(sid, factor):
    full_results["c_area"] = round(factor * full_results["w_area"], 2)
    new_c_areas = json.dumps(full_results["c_area"].tolist())
    await sio.emit("areaCalibrated", new_c_areas)

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



