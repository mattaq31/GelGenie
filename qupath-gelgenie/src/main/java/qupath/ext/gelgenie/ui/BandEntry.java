package qupath.ext.gelgenie.ui;

public class BandEntry {
    private String bandID;
    private int area;

    public BandEntry(String bandID, int area) {
        this.bandID = bandID;
        this.area = area;
    }

    public String getBandID() {
        return bandID;
    }

    public void setBandID(String bandID) {
        this.bandID = bandID;
    }

    public int getArea() {
        return area;
    }

    public void setArea(int area) {
        this.area = area;
    }
}
