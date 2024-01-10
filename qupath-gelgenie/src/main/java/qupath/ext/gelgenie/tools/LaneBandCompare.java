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

import java.util.Comparator;

public class LaneBandCompare implements Comparator<PathObject> {

    /**
     * Compares and sorts gel bands (PathObjects) based on their laneID and bandID.
     * @param a the first PathObject to be compared.
     * @param b the second PathObject to be compared.
     */
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
