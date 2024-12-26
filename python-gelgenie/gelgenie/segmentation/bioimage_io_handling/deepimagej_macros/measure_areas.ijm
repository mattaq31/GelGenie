// measures the area of all bands in the original image and exports to csv file

selectImage("Raw");
roiManager("Select", newArray(0, roiManager("count") - 1)); // Select all ROIs in the manager
roiManager("Add"); // Add them to the target image

// csv output titles
output = "Band_ID,Volume\n";

// Get the number of ROIs in the ROI Manager
roiCount = roiManager("count");
if (roiCount == 0) {
    print("No ROIs found in ROI Manager.");
    exit();
}

// Loop through each ROI
for (i = 0; i < roiCount; i++) {
    roiManager("Select", i);
    getStatistics(area, mean, min, max, std, hist); 
    sum = area * mean; // Calculate the sum of pixel values within the ROI (this isn't 100% accurate, but good enough...
    output += "" + i+1 + "," + sum + "\n"; // ROIs are displayed 1-indexed, but selected with a 0-index...
}

path = getDirectory("Choose a Directory") + "gel_band_measurements.csv";
File.saveString(output, path);
