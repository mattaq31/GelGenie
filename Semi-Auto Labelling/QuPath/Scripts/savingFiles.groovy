// select run for project to barch process all files


// change classifier name
classifierName = "Ivan-gels && Pair_4and5 Wk9 GelRed Save 2"

classifyDetectionsByCentroid(classifierName)

/**
 * Script to export image tiles (can be customized in various ways).
 */

// Get the current image (supports 'Run for project')
def imageData = getCurrentImageData()

// Define output path (here, relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'predictionImage', name)

mkdirs(pathOutput)

// string classifier, string path
writePredictionImage(classifierName, pathOutput + ".tif")   // Write tiles to the specified directory

// Done
print 'Done!'