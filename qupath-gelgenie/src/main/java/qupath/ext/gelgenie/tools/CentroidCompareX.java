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
