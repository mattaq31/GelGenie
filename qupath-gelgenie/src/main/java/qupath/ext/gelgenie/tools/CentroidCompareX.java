package qupath.ext.gelgenie.tools;

import qupath.lib.objects.PathObject;

import java.util.Comparator;

public class CentroidCompareX implements Comparator<PathObject> {
    public int compare(PathObject a, PathObject b) {
        if (a.getROI().getCentroidX() < b.getROI().getCentroidX())
            return -1;
        if (a.getROI().getCentroidX() == b.getROI().getCentroidX())
            return 0;
        return 1;
    }
}
