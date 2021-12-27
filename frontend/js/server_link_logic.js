// all communication with Python server occurs through socket 9111
const socket = io("http://localhost:9111");

// Initialize global variables and jquery selectors
$(document).ready(function () {
    window.curr_band_label = $("#band_label");
    window.curr_band_area = $("#band_area");
    window.curr_w_band_area = $("#w_band_area");
    window.curr_c_band_area = $("#c_band_area");
    window.removeMode = false;
});

// Load supplied image file and send to server for conversion to png
function previewFile() {
    let preview = document.querySelector('img'); // Get HTML img element
    let file = document.querySelector('input[type=file]').files[0]; // Get HTML file element (interface to allow user to select a file from disk)
    let reader = new FileReader(); // Create new file reader

    // When the user has chosen a file
    reader.onloadend = function () {
        let sourceImage = reader.result;
        // Emit a socket.io event which can be caught in Python, telling Python to read the image
        // This is because the prior code will not read .tif files correctly, and so we must use Python to convert the image to another format first
        socket.emit("imageToRead", sourceImage);
    }
    if (file) {
        reader.readAsDataURL(file); // Read image
        let load_img_cont = document.getElementById("load_img_cont");
        load_img_cont.style.display = "none";  // Hide the select image container
        second_screen.style.display = "block";   // Show the container with image
    } else {
        preview.src = "";
    }
}

// Listen for Python returning the source image in HTML displayable format
socket.on("sourceInPng", function (data) {
    let preview = document.getElementById('preview_img');
    preview.src = 'data:image/png;base64,'.concat(data["image"]);
    let lowerTh = data["otsu"];
    let higherTh = data["otsu"] + 25;
    document.getElementById("bg_value").value = parseInt(lowerTh);
    document.getElementById("fg_value").value = parseInt(higherTh);
    document.getElementById("out_foreground").value = parseInt(higherTh);
    document.getElementById("out_background").value = parseInt(lowerTh);
});

// routine to run after bands have been computed
socket.on("viewResult", function (data) {

    // Remove spinner from find bands button
    let spinnerHTML = '<span class="fas fa-search"></span>';
    $("#find_bands").html(spinnerHTML);

    // Show the image with found bands in the browser
    let preview = document.getElementById('preview_img');
    preview.src = 'data:image/png;base64,'.concat(data["file"]);

    // Decode variables into global scope TODO: how to reduce the mess here?
    window.band_centroids = JSON.parse(data["centroids"]);
    window.band_areas = JSON.parse(data["areas"]);
    window.band_w_areas = JSON.parse(data["w_areas"]);
    let band_bboxs = JSON.parse(data["bboxs"]);
    window.band_labels = JSON.parse(data["labels"]);
    let band_indices = JSON.parse(data["indices"]);
    window.band_c_areas = window.band_w_areas;

    window.base64_string = data["file"];  // global image overlay variables (light and dark)
    window.inv_base64_string = data["inverted_file"];

    let band_rectangles = $("#band_rectangles")

    // Find original image size
    let image_height = data["im_height"];
    let image_width = data["im_width"];
    window.image_height = image_height

    let size_magnitude = 1.0 // size of button relative to actual band

    // Create band rectangles for each band
    for (let i = 0; i < window.band_centroids.length; i++) {  // TODO: this system is prone to errors - it must be made more robust (use dictionaries instead of arrays)
        // Find width and height of each band in percent
        let band_height_100 = ((((band_bboxs[i][2] - band_bboxs[i][0]) * size_magnitude) / image_height) * 100).toString();
        let band_width_100 = ((((band_bboxs[i][3] - band_bboxs[i][1]) * size_magnitude) / image_width) * 100).toString();
        band_height_100 = band_height_100.concat("%");
        band_width_100 = band_width_100.concat("%");

        // Find the top left corner of each band in percent
        let band_y_100 = (band_bboxs[i][0] / image_height * 100).toString();
        band_y_100 = band_y_100.concat("%");
        let band_x_100 = (band_bboxs[i][1] / image_width * 100).toString();
        band_x_100 = band_x_100.concat("%");

        // Give each band a unique ID
        let current_id = "band_".concat(band_indices[i]);
        // Create the button with found width and height
        band_rectangles.append('<button type="button" id="' + current_id + '"  class="btn btn-outline-primary band-rect" style = "padding: 0; width:' + band_width_100 + '; height:' + band_height_100 + '"></button>');

        // Position the button
        let current_band = $("#" + current_id);
        current_band.css({top: band_y_100, left: band_x_100, position: 'absolute'});

        // Add event listener to button
        current_band.click(function () {
            if (window.removeMode) {
                removeClick(band_indices[i]);
            } else {
                bandClick(band_indices[i]);
            }
        });
    }
});

// When the user clicks on a band rectangle
function bandClick(bandNo) {

    let array_index = bandNo - 1;  // TODO: this implementation quirk needs to be fixed
    // Show the band details for this band in the band data table
    window.curr_band_label.html(window.band_labels[array_index]);
    window.curr_band_area.html(window.band_areas[array_index]);
    window.curr_w_band_area.html(window.band_w_areas[array_index]);
    window.curr_c_band_area.html(window.band_c_areas[array_index]);

    // Edit band labels
    let editBtn = $("#label-btn");
    editBtn.off(); // Removes the event listener from previous band, thus avoiding modifying label for all previously modified bands
    editBtn.click(function () {
        if (window.curr_band_label.attr('contenteditable') !== 'true') {
            window.curr_band_label.attr('contenteditable', 'true');
            editBtn.html('Save');
            window.curr_band_label.addClass("table-active");
        } else {
            window.curr_band_label.attr('contenteditable', 'false');
            // Change Button Text and Color
            editBtn.html('Edit');
            window.curr_band_label.removeClass("table-active");
            // Save the data to Python
            let updatedLabel = window.curr_band_label.html();
            // console.log(updatedLabel);
            socket.emit("updateBandLabel", bandNo, updatedLabel);
        }
    });

    // Find lane profile
    socket.emit("laneProfile", window.band_centroids[array_index][1], window.image_height);

    // Calibrate band volume
    let calibBtn = $("#calibrate_btn");

    calibBtn.click(function () {
        if (window.curr_c_band_area.attr('contenteditable') !== 'true') {
            window.curr_c_band_area.attr('contenteditable', 'true');
            calibBtn.html('Set');
            window.curr_c_band_area.addClass("table-active");
        } else {
            window.curr_c_band_area.attr('contenteditable', 'false');
            calibBtn.html('Edit');
            window.curr_c_band_area.removeClass("table-active");
            // Save the data to Python
            calArea = window.curr_c_band_area.html();
            let factor = calArea / window.band_w_areas[array_index]; // Set conversion factor between cal area and weighted area
            console.log("factor calculated");
            socket.emit("calibrateArea", factor);
        }
    });
}

$(document).ready(function () {

    // When the user selects the band finding parameters
    $("#find_bands_ready").click(function () {
        // Get the user selected parameters from sliders
        let sure_fg = $("#fg_value").val();
        let sure_bg = $("#bg_value").val();
        let repetitions = $("#repetitions").val();
        // Remove all previous band rectangles (if using find bands feature again, this erases previous results so as not to interfere)
        $(".band-rect").remove();
        // Call Python to find bands, passing parameters
        socket.emit("findBands", sure_fg, sure_bg, repetitions);
        // Hide modal
        $('#loading_bands').modal('hide');
        // Turn button into loading spinner
        let spinnerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span><span class="sr-only">Loading...</span>';
        $("#find_bands").html(spinnerHTML);
    });

    // Display provided lane profile
    socket.on("foundLaneProfile", function (b64String) {
        let laneProfile = document.getElementById('int_plot');
        laneProfile.src = 'data:image/png;base64,'.concat(b64String);
    });

    // When the band label is modified
    socket.on("labelUpdated", function (jsonLabels) {
        window.band_labels = JSON.parse(jsonLabels);
    });

    // When the band areas are calibrated
    socket.on("areaCalibrated", function (jsonCArea) {
        window.band_c_areas = JSON.parse(jsonCArea);
    });

    socket.on("imageUpdated", function (data) {
        // updates image with new results
        window.base64_string = data["non_inv"];
        window.inv_base64_string = data["inv"];
    });
});