package qupath.ext.gelgenie.ui;

import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.roi.interfaces.ROI;

import java.util.Objects;

/**
 * Main class that holds all data  displayed in gel data table.  Mainly consists of getters/setters.
 */
public class BandEntry extends PathObject {
    private int bandID;
    private int laneID;
    private String bandName;
    private double pixelCount;
    private double averageIntensity;
    private double rawVolume;
    private double globalVolume;
    private double localVolume;

    private double rollingVolume;
    private double normVolume = 0.0;
    private double normGlobal = 0.0;
    private double normLocal = 0.0;
    private double normRolling = 0.0;
    private PathObject parentAnnotation;

    public BandEntry(int bandID, int laneID, String bandName, double pixelCount, double averageIntensity, double rawVolume,
                     double globalVolume, double localVolume, double rollingVolume, PathObject parentAnnotation) {
        this.bandID = bandID;
        this.laneID = laneID;
        this.bandName = Objects.requireNonNullElse(bandName, "N/A");
        this.pixelCount = pixelCount;
        this.averageIntensity = averageIntensity;
        this.rawVolume = rawVolume;
        this.globalVolume = globalVolume;
        this.localVolume = localVolume;
        this.rollingVolume = rollingVolume;
        this.parentAnnotation = parentAnnotation;
    }

    public int getBandID() {
        return bandID;
    }

    public void setBandID(int bandID) {
        this.bandID = bandID;
    }

    public int getLaneID() { return laneID; }
    public void setLaneID(int laneID) {this.laneID = laneID;}

    public String getBandName() {
        return bandName;
    }

    public void setBandName(String bandName) {
        this.bandName = bandName;
    }

    public double getPixelCount() {
        return pixelCount;
    }

    public void setPixelCount(double pixelCount) {
        this.pixelCount = pixelCount;
    }

    public double getAverageIntensity() {
        return averageIntensity;
    }

    public void setAverageIntensity(double averageIntensity) {
        this.averageIntensity = averageIntensity;
    }

    public double getRawVolume() {
        return rawVolume;
    }

    public void setRawVolume(double rawVolume) {
        this.rawVolume = rawVolume;
    }

    public double getGlobalVolume() {
        return globalVolume;
    }

    public void setGlobalVolume(double globalVolume) {
        this.globalVolume = globalVolume;
    }

    public double getLocalVolume() {
        return localVolume;
    }

    public void setLocalVolume(double localVolume) {
        this.localVolume = localVolume;
    }

    public double getRollingVolume() {
        return rollingVolume;
    }

    public void setRollingVolume(double rollingVolume) {
        this.rollingVolume = rollingVolume;
    }

    public double getNormVolume() {
        return normVolume;
    }

    public void setNormVolume(double normVolume) {
        this.normVolume = normVolume;
    }

    public double getNormGlobal() {
        return normGlobal;
    }

    public void setNormGlobal(double normVolume) {
        this.normGlobal = normVolume;
    }

    public double getNormLocal() {
        return normLocal;
    }

    public void setNormLocal(double normVolume) {
        this.normLocal = normVolume;
    }

    public double getNormRolling() {
        return normRolling;
    }

    public void setNormRolling(double normRolling) {this.normRolling = normRolling; }

    public PathObject getParentAnnotation() {
        return parentAnnotation;
    }

    public void setParentAnnotation(PathObject parentAnnotation) {
        this.parentAnnotation = parentAnnotation;
    }

    // Methods below are required to implement the abstract methods in a PathObject.
    // However, these should never be used and the actual annotation PathObject should be access via getParentAnnotation().
    @Override
    public boolean isEditable() {
        throw new RuntimeException("This method has not been implemented in GelGenie.");
    }
    @Override
    public PathClass getPathClass() {
        throw new RuntimeException("This method has not been implemented in GelGenie.");
    }
    @Override
    public void setPathClass(PathClass pathClass, double classProbability) {
        throw new RuntimeException("This method has not been implemented in GelGenie.");
    }
    @Override
    public double getClassProbability() {
        throw new RuntimeException("This method has not been implemented in GelGenie.");
    }
    @Override
    public ROI getROI() {
        throw new RuntimeException("This method has not been implemented in GelGenie.");
    }
}
