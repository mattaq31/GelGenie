function percentile_normalization(){
	// this function was obtained from https://github.com/deepimagej/imagej-macros/blob/master/PercentileNormalization.ijm
	nBins = 1024; // the larger the more accurate
	getHistogram(values, counts, nBins);
	
	//create a cumulative histogram
	cumHist = newArray(nBins);
	cumHist[0] = values[0];
	for (i = 1; i < nBins; i++){ cumHist[i] = counts[i] + cumHist[i-1]; }
	
	//normalize the cumulative histogram
	normCumHist = newArray(nBins);
	for (i = 0; i < nBins; i++){  normCumHist[i] = cumHist[i]/
	cumHist[nBins-1]; }
		
	// find the 1th percentile (= 0.01)
	target = 0.01;
	i = 0;
	do {
	        i = i + 1;
	        // print("i=" + i + "  value=" + values[i] +  "  count=" + counts[i] + "cumHist= " + cumHist[i] + "  normCumHist= " + normCumHist[i] );
	} while (normCumHist[i] < target)
	mi = values[i];
	// print("1th percentile has value " + mi);
	
	// find the 99.9th percentile (= 0.999)
	target = 0.999;
	i = 0;
	do {
	        i = i + 1;
	        // print("i=" + i + "  value=" + values[i] +  "  count=" + counts[i] + "cumHist= " + cumHist[i] + "  normCumHist= " + normCumHist[i] );
	} while (normCumHist[i] < target)
	ma = values[i];
	// print("99.8th percentile has value " + ma);
	
	diff = ma-mi+1e-20; // add epsilon to avoid 0-divisions
	run("32-bit");
	run("Subtract...", "value="+mi);
	run("Divide...", "value="+diff);
	
}

// get input image
imgID=getImageID();
selectImage(imgID);

// duplicate to prepare input for model
run("Duplicate...", "title=Normalized-Input-Image"); 
selectWindow("Normalized-Input-Image");

getRawStatistics(count, mean, min, max, std);
print("Min pixel value:", min, "Max pixel value:", max);
if (bitDepth() == 8) {
    // For 8-bit images, divide by 255
    print("8-bit image detected. Normalizing by dividing by 255.");
    run("32-bit");
    run("Divide...", "value=255");
} else if (bitDepth() == 16) {
    // For 16-bit images, apply percentile normalization between 0.1% and 99.9%
    print("16-bit image detected. Applying percentile normalization.");
	percentile_normalization();

} else {
    print("Unsupported bit depth: " + bitDepth);
    exit();
}

print("Running model now...");

// model runs here
run("DeepImageJ Run", "modelPath=gelgenie_universal_model_bioimageio_09012025_231116 inputPath=null outputFolder=null displayOutput=all");

// close log to reduce clutter
selectWindow("Log");
run("Close");

// get model image (will have popped up last and so will be selected automatically)
modelImage = getTitle();  
selectImage(modelImage);

// Select the first channel
setSlice(1); // Channel 1 is slice 1
run("Duplicate...", "title=Channel_1");
selectImage(modelImage);
// Select the second channel
setSlice(2); // Channel 2 is slice 2
run("Duplicate...", "title=Channel_2");

// Calculate the difference and then threshold (basically performing the argmax operation necessary to combine the model's two outputs into one)
imageCalculator("Subtract create", "Channel_2", "Channel_1");
rename("Difference");
setThreshold(0, 99999999999);
run("Convert to Mask");
rename("Mask");

// cleans up
selectWindow("Channel_1");
run("Close");
selectWindow("Channel_2");
run("Close");

// partitions mask into different bands using particle analyzer tool
selectWindow("Mask");
run("Analyze Particles...", "  show=Overlay display clear add composite");

// goes back to original image, duplicates and adds overlay of all found bands
selectImage(imgID);
run("Duplicate...", "title=Image-Overlaid-with-Model-Mask"); 
selectWindow("Image-Overlaid-with-Model-Mask");
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

// save segmentation results
path = getDirectory("Choose a directory to save band areas") + "gel_band_measurements.csv";
File.saveString(output, path);

selectWindow("Mask");
run("Close");
selectWindow("Normalized-Input-Image");
run("Close");
selectImage(modelImage);
run("Close");