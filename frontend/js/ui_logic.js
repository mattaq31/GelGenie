window.curr_backgrnd = 'dark';  // TODO: this won't always be the case....

$(document).ready(function () {

    // Toggle background color
    $("#white_bg").click(function () {
        let preview = document.getElementById('preview_img');
        preview.src = 'data:image/png;base64,'.concat(inv_base64_string);
        window.curr_backgrnd = 'light';
    });
    $("#black_bg").click(function () {
        let preview = document.getElementById('preview_img');
        preview.src = 'data:image/png;base64,'.concat(base64_string);
        window.curr_backgrnd = 'dark';
    });

    // Initiate band finding
    $("#find_bands").click(function () {
        // Show modal with parameter selection
        $('#loading_bands').modal('show');
    });

    // If the user selects cancel on band finding modal
    $("#close_modal").click(function () {
        $('#loading_bands').modal('hide');
    });

    // Export to CSV
    $("#data_to_csv").click(function () {
        window.electron.export('main_data_csv')
    });

    // Export image with found bands
    $("#image_to_tif").click(function () {
        if (window.curr_backgrnd === 'light') {
            window.electron.export('main_image_light');
        }
        else {
            window.electron.export('main_image_dark');
        }
    });

    // Export current profile to CSV
    $("#profile_to_csv").click(function () {
        window.electron.export('profile_csv')
    });

    // Export current profile to image output
    $("#profile_to_tif").click(function () {
        window.electron.export('profile_image')
    });

});