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

imgID=getImageID();
selectImage(imgID);

run("Duplicate...", "title=Raw Image"); // Duplicates the original image to avoid modifying the original
selectWindow("Raw");

// padding to multiple of 32 to satisfy U-Net conditions
width = getWidth();
height = getHeight();

if (height % 32 == 0) {
	newHeight = height;
}else {
	newHeight = 32 * floor(height / 32 + 1);
}

if (width % 32 == 0) {
	newWidth = width;
}else {
	newWidth = 32 * floor(width / 32 + 1);
}

run("Canvas Size...", "width="+newWidth+" height="+newHeight+" position=Center zero");

run("Duplicate...", "title=Normalized Image"); 
selectWindow("Normalized");
getRawStatistics(count, mean, min, max, std);
print("Min:", min, "Max:", max);
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
}
selectWindow("Log");
run("Close");

// ideally, the deep model would be scriptable at this point, but there's some bug preventing DIJ from finding the current image when scripting....
//run("DeepImageJ Run", "modelPath=packaged_model_24122024_113534 inputPath=/Users/matt/Desktop/32.tif outputFolder=/Users/matt/Desktop/dij_output displayOutput=0")
