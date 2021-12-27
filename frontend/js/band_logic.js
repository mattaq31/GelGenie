function removeClick(bandNo) {  // TODO: band removal should also remove colour in image....
    // TODO: need to split underlying image and segmentation overlay from each other
    // Get band ID
    current_id = "#band_".concat(bandNo);
    $(current_id).remove();
    socket.emit("removeBand", bandNo);
}

$(document).ready(function () {
    // Go into remove bands mode
    $("#remove_bands").click(function () {
        let bandrect = $(".band-rect")
        if (window.removeMode === false) {
            bandrect.removeClass("btn-outline-primary");
            bandrect.addClass("btn-outline-danger");
            window.removeMode = true;
        } else {
            bandrect.removeClass("btn-outline-danger");
            bandrect.addClass("btn-outline-primary");
            window.removeMode = false;
        }
    });
});