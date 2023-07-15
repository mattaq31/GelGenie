createSelectAllObject(true)
// TODO: CHANGE FOLLOWING LINE TO WHAT YOU HAVE FOR YOUR CELL DETECTION
runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImage": "Red",  "backgroundRadius": 15.0,  "medianRadius": 0.0,  "sigma": 3.0,  "minArea": 10.0,  "maxArea": 1000.0,  "threshold": 25.0,  "watershedPostProcess": true,  "cellExpansion": 5.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');

def anns = []
def meanIntensity = 0
getCellObjects().forEach {
    anns.add(PathObjects.createAnnotationObject(it.getNucleusROI(), it.getPathClass()))
    meanIntensity += (it.getMeasurementList().getMeasurementValue("Nucleus: Red mean"))    // CHANGE TO YOUR CHANNEL NAME
}

printf("Mean intensity of detected objects: %f", meanIntensity/getCellObjects().size())
addObjects(anns)
clearDetections()
print "Done!"