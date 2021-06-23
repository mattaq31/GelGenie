// Initialize tooltips
$(document).ready(function() {
    $('.open').tooltip({trigger: 'hover', title: 'Open image', placement: 'bottom'});
    $('.find').tooltip({trigger: 'hover', title: 'Find bands', placement: 'bottom'});
    // TO DO: Add the rest of the tooltips
});

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
    
    var loadImgCont = document.getElementById("loadImgCont");
    loadImgCont.style.display = "none";  // Hide the select image container
    loadedImg.style.display = "block";   // Show the container with image
  } else {
    preview.src = "";
  }
}


