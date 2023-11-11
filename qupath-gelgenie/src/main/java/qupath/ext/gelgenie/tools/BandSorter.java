package qupath.ext.gelgenie.tools;

import qupath.lib.objects.PathObject;
import java.util.*;

public class BandSorter {

    public static void  LabelBands(Collection<PathObject> bands){
        int bandIdCounter;
        int laneIdCounter = 1;

        while (!bands.isEmpty()){
            // TODO: sometimes bands are still missed with this system.
            //  Will probably have to recursively repeat the search for each new band that is added to a lane.
            // Or implement someway to identify and delete split/spurious bands.
            PathObject leftMostBand = Collections.min(bands, new centroidCompareX());
            double lowerX = leftMostBand.getROI().getBoundsX();
            double upperX = leftMostBand.getROI().getBoundsX() + leftMostBand.getROI().getBoundsWidth();
            ArrayList<PathObject> currentLaneBands = new ArrayList<>();
            for (PathObject band: bands) {
                double xW = band.getROI().getBoundsWidth();
                double xLow = band.getROI().getBoundsX();
                double xHigh = xLow + xW;
                double xCent = band.getROI().getCentroidX();

                if (xCent < upperX && xCent > lowerX) {
                    currentLaneBands.add(band);
                } else if (xLow < upperX && xHigh > upperX) {
                    currentLaneBands.add(band);
                } else if (xLow < lowerX && xHigh > lowerX) {
                    currentLaneBands.add(band);
                }
            }
            currentLaneBands.sort(Comparator.comparing((PathObject p) -> p.getROI().getCentroidY()));
            bandIdCounter = 1;
            for (PathObject band: currentLaneBands) {
                band.setName(String.format("L%d-%d", laneIdCounter, bandIdCounter));
                band.getMeasurementList().put("LaneID", laneIdCounter);
                band.getMeasurementList().put("BandID", bandIdCounter);
                bandIdCounter++;
                bands.remove(band);
            }

            laneIdCounter++;
        }
    }
}

