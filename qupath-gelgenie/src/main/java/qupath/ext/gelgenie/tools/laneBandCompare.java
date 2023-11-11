package qupath.ext.gelgenie.tools;

import qupath.lib.objects.PathObject;

import java.util.Comparator;

public class laneBandCompare implements Comparator<PathObject> {
    public int compare(PathObject a, PathObject b) {

        // defaults just in case lane/band ID are not set
        if (a.getMeasurements().get("LaneID") == null){
            return 1;
        }
        if (b.getMeasurements().get("LaneID") == null){
            return -1;
        }

        if (a.getMeasurements().get("BandID") == null){
            return 1;
        }
        if (b.getMeasurements().get("BandID") == null){
            return -1;
        }

        int laneA = a.getMeasurements().get("LaneID").intValue();
        int laneB = b.getMeasurements().get("LaneID").intValue();
        int bandA = a.getMeasurements().get("BandID").intValue();
        int bandB = b.getMeasurements().get("BandID").intValue();

        if (laneA < laneB) // lane ID takes priority
            return -1;
        if (laneA == laneB){
            return Double.compare(bandA, bandB);
        }
        return 1;
    }
}
