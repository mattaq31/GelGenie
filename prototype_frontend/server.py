from aiohttp import web
import socketio

import os
import base64
from io import BytesIO
import json
import jsonpickle

import cv2
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

from gelgenie.gel_tools import GelAnalysis

# Allows one to run server from any base filepath
os.chdir(os.path.abspath(os.path.join(__file__, os.path.pardir)))

# Create an async aiohttp server
sio = socketio.AsyncServer(cors_allowed_origins='*', max_http_buffer_size=(50*1000*1000))  # 50 MB max buffer size (very imp)
app = web.Application()
sio.attach(app)

gel_analysis = GelAnalysis()  # main gel analysis object
profile_image = None  # temp global variable TODO: replace


@sio.on("imageToRead")
async def imageToRead(sid, source_b64):
    """
    Frontend-link function.
    Takes a user-selected image from JS, converts it to png and also finds otsu threshold value.
    :param sid: Call SID
    :param source_b64: Image encoded in base64 format.
    :return: PNG image and estimated otsu percentage.
    """
    global gel_analysis
    # Load image from b64 received from JS, giving width and height
    gel_analysis.set_image(image=source_b64.split('base64,')[-1], image_type='b64')

    # Encode original image to b64 for frontend display (must be .png for browsers to be able to display)
    png_image = gel_analysis.get_b64_image()

    # Get otsu threshold value to estimate band finding parameters
    otsu_percent = gel_analysis.get_otsu_threshold()

    await sio.emit('sourceInPng', {"image": png_image, "otsu": otsu_percent})


@sio.on("findBands")
async def findBands(sid, sure_fg, sure_bg, repetitions):
    """Finds bands on an image with user-defined parameters"""
    global gel_analysis

    # Find bands using watershed segmentation
    gel_analysis.find_bands(sure_fg, sure_bg, repetitions)

    # obtain overlay images for frontend display
    new_image_b64, inverted_image_b64 = gel_analysis.get_bands_overlay_b64()

    # Encode selected band props to json
    encoded_centroids = json.dumps(gel_analysis.band_dict['center'])
    encoded_areas = json.dumps(gel_analysis.band_dict['area'])
    encoded_w_areas = json.dumps(gel_analysis.band_dict['w_area'])
    band_props = jsonpickle.encode(gel_analysis.band_regions)
    encoded_bboxs = json.dumps(gel_analysis.band_dict['bbox'])
    encoded_indices = json.dumps(gel_analysis.band_dict['id'])
    encoded_labels = json.dumps(gel_analysis.band_dict['label'])
    # Send image and band props to JS
    await sio.emit('viewResult', {'file': new_image_b64, 'inverted_file': inverted_image_b64,
                                  'im_width': gel_analysis.base_image.width,
                                  'im_height': gel_analysis.base_image.height,
                                  'props': band_props,
                                  'centroids': encoded_centroids, 'bboxs': encoded_bboxs,
                                  'areas': encoded_areas, 'w_areas': encoded_w_areas,
                                  'indices': encoded_indices, 'labels': encoded_labels})


@sio.on("updateBandLabel")
async def updateBandLabel(sid, band_id, updated_label):
    """Updates band label of a single band"""
    global gel_analysis

    gel_analysis.band_dataframe.loc[band_id, "label"] = updated_label

    print("The label of band %s has been changed to %s" % (band_id, updated_label))

    new_labels = json.dumps(gel_analysis.band_dataframe["label"].tolist())
    await sio.emit("labelUpdated", new_labels)


@sio.on("laneProfile")
async def laneProfile(sid, x_pos, y_end):
    """Draws and returns lane profile of selected band."""
    global gel_analysis, profile_image

    profile = gel_analysis.extract_lane_profile(x_pos, y_end)
    fig = plt.figure(figsize=(4, 11), facecolor='black')
    plt.plot(profile, range(len(profile)))
    plt.gca().invert_yaxis()
    plt.axis("off")
    my_stringIObytes = BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('utf-8')

    profile_image = my_stringIObytes
    plt.close(fig)
    await sio.emit('foundLaneProfile', my_base64_jpgData)


@sio.on("calibrateArea")
async def calibrateArea(sid, factor):
    """Sets calibration factor for area."""
    global gel_analysis

    gel_analysis.calibrate_band_area(factor)
    new_c_areas = json.dumps(gel_analysis.band_dataframe['c_area'].tolist())
    await sio.emit("areaCalibrated", new_c_areas)

@sio.on("removeBand")
async def removeBand(sid, band_no):
    """Removes a band from everything."""
    global gel_analysis

    # Remove band from image
    gel_analysis.remove_band(band_no)

    new_image_b64, inverted_image_b64 = gel_analysis.get_bands_overlay_b64()

    await sio.emit("imageUpdated", {"non_inv": new_image_b64, "inv": inverted_image_b64})


@sio.on("exportToCsv")
async def exportToCsv(sid, csv_loc):
    """Exports band data to .csv format."""
    global gel_analysis
    # Export pandas table to csv
    gel_analysis.band_dataframe.to_csv(csv_loc)


@sio.on("exportToBandImage")
async def exportToBandImage(sid, output_file, inverted):
    """Exports found band images"""
    global gel_analysis
    if inverted:
        output_im = gel_analysis.overlay_inverted
    else:
        output_im = gel_analysis.overlayed_image_bands

    output_im = np.uint16(output_im * 65535)  # TODO: remove harcoding
    cv2.imwrite(output_file, output_im)


@sio.on("exportProfileGraph")
async def exportProfileGraph(sid, output_file):
    """Exports lane profile in image/graph form for use in a report/paper."""

    global profile_image

    with open(output_file, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(profile_image.getbuffer())


@sio.on("exportProfileCsv")
async def exportProfileCsv(sid, csv_loc):
    """Exports lane profile in csv form for further use in another program."""

    global gel_analysis
    pd_profile = pd.DataFrame(gel_analysis.current_profile)
    pd_profile.to_csv(csv_loc)


@sio.on("ping")
async def ping(sid):
    print('Pinged with SID:', sid)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=9111)
