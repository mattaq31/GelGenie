// Initialize the socket.io connection on port 8080
const socket = io("http://localhost:9111");

// Load file
function previewFile() {
    var preview = document.querySelector('img'); // Get HTML img element
    var file = document.querySelector('input[type=file]').files[0]; // Get HTML file element (interface to allow user to select a file from disk)
    var reader = new FileReader(); // Create new file reader

    // When the user has chosen a file
    reader.onloadend = function () {
        // Set the chosen image to be displayed
        let sourceImage = reader.result;
        preview.src = sourceImage;
        // console.log(sourceImage.substr(0)) TODO: the substring cannot be hard-coded!
        // Get the base64 representation of the chosen image by removing initial string that is identical across images
        let base64Img = sourceImage.substr(22);
        // console.log(base64Img);
        // Emit a socket.io event which can be caught in Python, telling Python to read the image
        // This is because the prior code will not read .tif files correctly, and so we must use Python to convert the image to another format first
        socket.emit("imageToRead", base64Img);
    }
    if (file) {
        reader.readAsDataURL(file); // Read image
        var load_img_cont = document.getElementById("load_img_cont");
        load_img_cont.style.display = "none";  // Hide the select image container
        second_screen.style.display = "block";   // Show the container with image
    } else {
        preview.src = "";
    }
}

// Listen for Python returning the source image in HTML displayable format
socket.on("sourceInPng", function (data) {
    var preview = document.getElementById('preview_img');
    preview.src = 'data:image/png;base64,'.concat(data["image"]);
    original_b64 = data["image"];
    // console.log(original_b64);
    var lowerTh = data["otsu"];
    var higherTh = data["otsu"] + 25;
    document.getElementById("bg_value").value = lowerTh;
    document.getElementById("fg_value").value = higherTh;

});


// Initialize variables
var image_height;
inv_base64_string = "";
removeMode = false;

// When bands have been found
socket.on("viewResult", function (data) {

    // Remove spinner from find bands button
    let spinnerHTML = '<span class="fas fa-search"></span>';
    $("#find_bands").html(spinnerHTML);

    // Show the image with found bands in the browser
    var preview = document.getElementById('preview_img');
    preview.src = 'data:image/png;base64,'.concat(data["file"]);

    // Decode variables
    band_props = JSON.parse(data.props);
    band_centroids = JSON.parse(data.centroids);
    band_areas = JSON.parse(data.areas);
    band_w_areas = JSON.parse(data.w_areas);
    band_bboxs = JSON.parse(data.bboxs);
    band_labels = JSON.parse(data.labels);
    band_indices = JSON.parse(data.indices);
    band_c_areas = band_w_areas;
    // console.log(band_props);
    base64_string = data["file"];
    inv_base64_string = data["inverted_file"];
    // console.log(data["file"]);

    // Position band rectangle div in same position as displayed image (to make sure bands display in proper positions)
    let actual_im_height = $("#preview_img").css("height");
    let actual_im_width = $("#preview_img").css("width");
    $("#band_rectangles").height(actual_im_height);
    $("#band_rectangles").width(actual_im_width);
    let im_position = $("#preview_img").position();
    $("#band_rectangles").css({top: im_position.top, left: im_position.left, position: 'absolute'});

    // Find original image size
    image_height = data.im_height;
    // console.log(image_height);
    image_width = data.im_width;
    // console.log(image_width);

    // Create band rectangles for each band
    for (let i = 0; i < band_centroids.length; i++) {

        // Find width and height of each band in percent
        let band_height_100 = ((band_bboxs[i][2] - band_bboxs[i][0]) * 0.8 / image_height * 100).toString();
        let band_width_100 = ((band_bboxs[i][3] - band_bboxs[i][1]) * 0.8 / image_height * 100).toString();
        band_height_100 = band_height_100.concat("%");
        band_width_100 = band_width_100.concat("%");

        // Find the top left corner of each band in percent
        let band_y_100 = (band_bboxs[i][0] / image_height * 100).toString();
        band_y_100 = band_y_100.concat("%");
        let band_x_100 = (band_bboxs[i][1] / image_width * 100).toString();
        band_x_100 = band_x_100.concat("%");
        console.log(band_indices)
        // Give each band a unique ID
        current_id = "band_".concat(band_indices[i]);
        // Create the button with found width and height
        $("#band_rectangles").append('<button type="button" id="' + current_id + '"  class="btn btn-outline-primary band-rect" style = "width:' + band_width_100 + '; height:' + band_height_100 + '"></button>');
        // Position the button
        $("#" + current_id).css({top: band_y_100, left: band_x_100, position: 'absolute'});
        // Add event listener to button
        $("#" + current_id).click(function () {
            if (removeMode) {
                console.log(band_indices[i])
                removeClick(band_indices[i]);
            } else {
                bandClick(band_indices[i]);
            }
        });
    }
});

// When the user clicks on a band rectangle
function bandClick(bandNo) {

    // console.log("The band area is " + band_areas[bandNo]);
    // console.log("The band label is " + band_labels[bandNo]);

    // Show the band details for this band in the band data table
    $("#band_label").html(band_labels[bandNo]);
    $("#band_area").html(band_areas[bandNo]);
    $("#w_band_area").html(band_w_areas[bandNo]);
    $("#c_band_area").html(band_c_areas[bandNo]);

    // Edit band labels
    let bandLabel = $("#band_label");
    let editBtn = $("#label-btn");
    editBtn.off(); // Removes the event listener from previous band, thus avoiding modifying label for all previously modified bands
    editBtn.click(function () {
        if (bandLabel.attr('contenteditable') !== 'true') {
            bandLabel.attr('contenteditable', 'true');
            editBtn.html('Save');
            bandLabel.addClass("table-active");
        } else {
            bandLabel.attr('contenteditable', 'false');
            // Change Button Text and Color
            editBtn.html('Edit');
            bandLabel.removeClass("table-active");
            // Save the data to Python
            updatedLabel = bandLabel.html();
            // console.log(updatedLabel);
            socket.emit("updateBandLabel", (bandNo + 1), updatedLabel); // Bands start at 0 in JS but at 1 in the Pandas table
        }
    });

    // Find lane profile
    socket.emit("laneProfile", band_centroids[bandNo][1], image_height);

    // Export image with lane profile of single selected lane
    $("#profile_to_tif").click(function () {
        socket.emit("exportProfileGraph", bandNo);
    });

    // Export csv with lane profile of single selected lane
    $("#profile_to_csv").click(function () {
        socket.emit("exportProfileCsv", bandNo);
    });

    // Calibrate band volume
    let calAreaBox = $("#c_band_area");
    let calibBtn = $("#calibrate_btn");
    calibBtn.click(function () {
        if (calAreaBox.attr('contenteditable') !== 'true') {
            calAreaBox.attr('contenteditable', 'true');
            calibBtn.html('Set');
            calAreaBox.addClass("table-active");
        } else {
            calAreaBox.attr('contenteditable', 'false');
            calibBtn.html('Edit');
            calAreaBox.removeClass("table-active");
            // Save the data to Python
            calArea = calAreaBox.html();
            let factor = calArea / band_w_areas[bandNo]; // Set conversion factor between cal area and weighted area
            console.log("factor calculated");
            socket.emit("calibrateArea", factor);
        }
    });
}


function removeClick(bandNo) {
    // Get band ID
    console.log(bandNo)
    current_id = "#band_".concat(bandNo);
    $(current_id).remove();
    socket.emit("removeBand", bandNo);
}


// Toolbar buttons
$(document).ready(function () {

    // Toggle background color
    $("#white_bg").click(function () {
        var preview = document.getElementById('preview_img');
        preview.src = 'data:image/png;base64,'.concat(inv_base64_string);
    });
    $("#black_bg").click(function () {
        var preview = document.getElementById('preview_img');
        preview.src = 'data:image/png;base64,'.concat(base64_string);
    });

    // Initiate band finding
    $("#find_bands").click(function () {
        // socket.emit("findBands", original_b64);
        // Show modal with parameter selection
        $('#loading_bands').modal('show');
    });

    // When the user selects the band finding parameters
    $("#find_bands_ready").click(function () {
        // Get the user selected parameters from sliders
        let sure_fg = $("#fg_value").val();
        let sure_bg = $("#bg_value").val();
        let repetitions = $("#repetitions").val();
        // Remove all previous band rectangles (if using find bands feature again, this erases previous results so as not to interfere)
        $(".band-rect").remove();
        // Call Python to find bands, passing parameters
        socket.emit("findBands", original_b64, sure_fg, sure_bg, repetitions);
        // Hide modal
        $('#loading_bands').modal('hide');
        // Turn button into loading spinner
        let spinnerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span><span class="sr-only">Loading...</span>';
        $("#find_bands").html(spinnerHTML);
    });

    // If the user selects cancel on band finding modal
    $("#close_modal").click(function () {
        $('#loading_bands').modal('hide');
    });


    // Export to CSV
    $("#data_to_csv").click(function () {
        socket.emit("exportToCsv");
    });

    // Export image with found bands
    $("#image_to_tif").click(function () {
        socket.emit("exportToBandImage");
    });

    // Find lane profile
    socket.on("foundLaneProfile", function (b64String) {
        var laneProfile = document.getElementById('int_plot');
        laneProfile.src = 'data:image/png;base64,'.concat(b64String);
    });

    // When the band label is modified
    socket.on("labelUpdated", function (jsonLabels) {
        band_labels = JSON.parse(jsonLabels);
    });

    // When the band areas are calibrated
    socket.on("areaCalibrated", function (jsonCArea) {
        band_c_areas = JSON.parse(jsonCArea);
    });

    // Go into remove bands mode
    $("#remove_bands").click(function () {
        if (removeMode == false) {
            $(".band-rect").removeClass("btn-outline-primary");
            $(".band-rect").addClass("btn-outline-danger");
            removeMode = true;
        } else {
            $(".band-rect").removeClass("btn-outline-danger");
            $(".band-rect").addClass("btn-outline-primary");
            removeMode = false;
        }
    });

    socket.on("imageUpdated", function (data) {
        base64_string = data["non_inv"];
        inv_base64_string = data["inv"];
    });

});