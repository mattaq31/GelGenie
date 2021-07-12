

// Initialize tooltips
$(document).ready(function() {
    $('.open').tooltip({trigger: 'hover', title: 'Open image', placement: 'bottom'});
    $('.find').tooltip({trigger: 'hover', title: 'Find bands', placement: 'bottom'});
    // TO DO: Add the rest of the tooltips
});






/*
// Open the image
function previewFile() {
  var preview = document.querySelector('img');
  var file    = document.querySelector('input[type=file]').files[0];
  var reader  = new FileReader();

  reader.onloadend = function () {
    preview.src = reader.result;
  }

  if (file) {
    reader.readAsDataURL(file);
    socket.emit("imageToRead", file);
    var loadImgCont = document.getElementById("loadImgCont");
    loadImgCont.style.display = "none";  // Hide the select image container
    loadedImg.style.display = "block";   // Show the container with image
  } else {
    preview.src = "";
  }
}
*/

/*
// Socket io
const app = require('express')();
const http = require('http').Server(app);
const io = require('socket.io')(http, {
  allowEIO3: true // false by default
});
const port = process.env.PORT || 3000;

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

io.on('connection', (socket) => {
  socket.on('chat message', msg => {
    io.emit('chat message', msg);
  });
});

http.listen(port, () => {
  console.log(`Socket.IO server running at http://localhost:${port}/`);
});
*/
