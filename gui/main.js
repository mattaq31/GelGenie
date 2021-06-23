// Initialize tooltips
$(document).ready(function() {
    $('.open').tooltip({trigger: 'hover', title: 'Open image', placement: 'bottom'});
    $('.find').tooltip({trigger: 'hover', title: 'Find bands', placement: 'bottom'});
});

// Open an image when button is clicked
$('.open').on('click', function () { /* do stuff */ });