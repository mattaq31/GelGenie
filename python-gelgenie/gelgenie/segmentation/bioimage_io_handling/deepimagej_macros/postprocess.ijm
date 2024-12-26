// make sure to select the model output first here...
imageTitle = getTitle();  
selectImage(imageTitle);
// Select the first channel
setSlice(1); // Channel 1 is slice 1
run("Duplicate...", "title=Channel_1");
selectImage(imageTitle);
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
