from aiohttp import web
import socketio

# Create an async aiohttp server
sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

# Define aiohttp endpoints
# This will deliver the main.html file to the client once connected.
async def index(request):
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