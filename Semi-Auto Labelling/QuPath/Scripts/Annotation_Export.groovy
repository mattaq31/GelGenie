import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'Annotaton_Export', name)
mkdirs(pathOutput)

// Define output resolution
double requestedPixelSize = 2.0

// Convert to downsample
double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('Gel Band', 1)      // Choose output labels (the order matters!)
    // .addLabel('Stroma', 2)
    // .addLabel('Other', 3)
    .lineThickness(2)          // Optionally export annotation boundaries with another label
    //.setBoundaryLabel('Boundary*', 4) // Define annotation boundary label
    .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
    .build()


// Export each region
int i = 0
for (annotation in getAnnotationObjects()) {
    def region = RegionRequest.createInstance(
        labelServer.getPath(), downsample, annotation.getROI())
    i++
    def outputPath = buildFilePath(pathOutput, 'Region ' + i + '.png')
    writeImageRegion(labelServer, region, outputPath)
}