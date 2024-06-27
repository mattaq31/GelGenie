/**
 * Copyright 2024 University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package qupath.ext.gelgenie.tools;

import qupath.lib.objects.PathObject;
import qupath.lib.scripting.QP;

import java.util.*;

public class BandSorter {

    public static void  LabelBands(Collection<PathObject> bands){
        int bandIdCounter;
        int laneIdCounter = 1;

        while (!bands.isEmpty()){
            // TODO: sometimes bands are still missed with this system.
            //  Will probably have to recursively repeat the search for each new band that is added to a lane.
            // Or implement someway to identify and delete split/spurious bands.
            PathObject leftMostBand = Collections.min(bands, new CentroidCompareX());
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

    /**
     * Script-friendly system where labelling is applied to all gel bands in the current image.
     */
    public static void LabelBands(){
        Collection<PathObject> actionableAnnotations = new ArrayList<>();
        for (PathObject annot : QP.getAnnotationObjects()) {
            if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Gel Band")) {
                actionableAnnotations.add(annot);
            }
        }
        LabelBands(actionableAnnotations);
    }
}

