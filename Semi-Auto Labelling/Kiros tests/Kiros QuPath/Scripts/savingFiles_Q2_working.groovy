// select run for project to barch process all files


// change classifier name
classifierName = "UVP01961May212019_(1)"

classifyDetectionsByCentroid(classifierName)

/**
 * Script to export image tiles (can be customized in various ways).
 */

// Get the current image (supports 'Run for project')
def imageData = getCurrentImageData()

// Define output path (here, relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'predictionImage', name)

mkdirs(PROJECT_BASE_DIR + 'predictionImage')

// string classifier, string path
writePredictionImage(classifierName, pathOutput + ".tif")   // Write tiles to the specified directory

// Done
print 'Done!'