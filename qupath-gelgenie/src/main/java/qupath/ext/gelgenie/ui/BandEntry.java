package qupath.ext.gelgenie.ui;

import qupath.lib.objects.PathObject;

import javafx.scene.image.ImageView;

public class BandEntry {
    private int bandID;
    private String bandName;
    private double pixelCount;
    private double averageIntensity;
    private double rawVolume;
    private double globalVolume;
    private double localVolume;
    private double normVolume;
    private ImageView thumbnail;

    public BandEntry(int bandID, String bandName, double pixelCount, double averageIntensity, double rawVolume,
                     double globalVolume, double localVolume, double normVolume) {
        this.bandID = bandID;
        this.bandName = bandName;
        this.pixelCount = pixelCount;
        this.averageIntensity = averageIntensity;
        this.rawVolume = rawVolume;
        this.globalVolume = globalVolume;
        this.localVolume = localVolume;
        this.normVolume = normVolume;
    }
    public BandEntry(int bandID, String bandName, double pixelCount, double averageIntensity, double rawVolume,
                     double globalVolume, double localVolume, double normVolume, ImageView thumbnail) {
        this.bandID = bandID;
        this.bandName = bandName;
        this.pixelCount = pixelCount;
        this.averageIntensity = averageIntensity;
        this.rawVolume = rawVolume;
        this.globalVolume = globalVolume;
        this.localVolume = localVolume;
        this.normVolume = normVolume;
        this.thumbnail = thumbnail;
    }

    public int getBandID() {
        return bandID;
    }

    public void setBandID(int bandID) {
        this.bandID = bandID;
    }

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

    public ImageView getThumbnail() {
        return thumbnail;
    }

    public void setThumbnail(ImageView thumbnail) {
        this.thumbnail = thumbnail;
    }
}
