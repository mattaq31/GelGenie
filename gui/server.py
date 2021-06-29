from aiohttp import web
import socketio
import os
from lane_finder import LaneFinder
import cv2

# allows one to run server from any base filepath
os.chdir(os.path.abspath(os.path.join(__file__, os.path.pardir)))


# Create an async aiohttp server
sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

@sio.on("imageToRead")
def imageToRead(sid, file):
    print("Finding bands in ", file)
    image = LaneFinder(cv2.imread(file))
    result = image.find_lanes()
    cv2.imwrite("results/result1.jpg", result)
    sio.emit('viewResult', {'data': 'foobar'})

# Define aiohttp endpoints
# This will deliver the main.html file to the client once connected.
async def index(request):
    print("request")
    with open('main.html') as f:
        return web.Response(text=f.read(), content_type='html')

# Bind our aiohttp endpoint to our app router
app.router.add_get('/', index)

# Add the static css and js files to the app router
app.router.add_static('/', 'static')

# We kick off our server
if __name__ == '__main__':
    web.run_app(app)

async def button_find_bands(request):
    return web.Response()



