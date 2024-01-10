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

public class CentroidCompareX implements Comparator<PathObject> {

    /**
     * Compares two PathObjects based on their X centroid coordinates.
     * @param a the first PathObject to be compared.
     * @param b the second PathObject to be compared.
     * @return -1 if the X centroid of a is less than the X centroid of b, 0 if they are equal,
     * and 1 if the X centroid of a is greater than the X centroid of b.
     */
    public int compare(PathObject a, PathObject b) {
        if (a.getROI().getCentroidX() < b.getROI().getCentroidX())
            return -1;
        if (a.getROI().getCentroidX() == b.getROI().getCentroidX())
            return 0;
        return 1;
    }
}
