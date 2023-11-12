package qupath.ext.gelgenie.ui;

import javafx.scene.image.ImageView;
import qupath.lib.objects.PathObject;

import java.util.Objects;

/**
 * Main class that holds all data  displayed in gel data table.  Mainly consists of getters/setters.
 */
public class BandEntry {
    private int bandID;
    private int laneID;
    private String bandName;
    private double pixelCount;
    private double averageIntensity;
    private double rawVolume;
    private double globalVolume;
    private double localVolume;
    private double normVolume;
    private PathObject parentAnnotation;

    public BandEntry(int bandID, int laneID, String bandName, double pixelCount, double averageIntensity, double rawVolume,
                     double globalVolume, double localVolume, double normVolume) {
        this.bandID = bandID;
        this.laneID = laneID;
        this.bandName = bandName;
        this.pixelCount = pixelCount;
        this.averageIntensity = averageIntensity;
        this.rawVolume = rawVolume;
        this.globalVolume = globalVolume;
        this.localVolume = localVolume;
        this.normVolume = normVolume;
    }

    public BandEntry(int bandID, int laneID, String bandName, double pixelCount, double averageIntensity, double rawVolume,
                     double globalVolume, double localVolume, double normVolume, PathObject parentAnnotation) {
        this.bandID = bandID;
        this.laneID = laneID;
        this.bandName = Objects.requireNonNullElse(bandName, "N/A");
        this.pixelCount = pixelCount;
        this.averageIntensity = averageIntensity;
        this.rawVolume = rawVolume;
        this.globalVolume = globalVolume;
        this.localVolume = localVolume;
        this.normVolume = normVolume;
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

    public double getNormVolume() {
        return normVolume;
    }

    public void setNormVolume(double normVolume) {
        this.normVolume = normVolume;
    }

    public PathObject getParentAnnotation() {
        return parentAnnotation;
    }

    public void setParentAnnotation(PathObject parentAnnotation) {
        this.parentAnnotation = parentAnnotation;
    }
}
