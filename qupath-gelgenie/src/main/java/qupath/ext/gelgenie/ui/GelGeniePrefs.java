package qupath.ext.gelgenie.ui;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.Property;
import javafx.beans.property.StringProperty;
import qupath.lib.gui.prefs.PathPrefs;

public class GelGeniePrefs {
    private static final StringProperty deviceProperty = PathPrefs.createPersistentPreference("gelgenie.device", "cpu");
    private static final BooleanProperty useDJLProperty = PathPrefs.createPersistentPreference("gelgenie.djl", false);
    private static final BooleanProperty deletePreviousBandsProperty = PathPrefs.createPersistentPreference("gelgenie.deletepreviousbands", false);
    private static final BooleanProperty globalCorrectionProperty = PathPrefs.createPersistentPreference("gelgenie.globalcorrection", false);
    private static final BooleanProperty localCorrectionProperty = PathPrefs.createPersistentPreference("gelgenie.localcorrection", false);
    private static final BooleanProperty rollingCorrectionProperty = PathPrefs.createPersistentPreference("gelgenie.rollingcorrection", false);

    private static final Property<Integer> localCorrectionPixels = PathPrefs.createPersistentPreference("gelgenie.localcorrectionpixels", 5).asObject();
    private static final Property<Integer> rollingRadius = PathPrefs.createPersistentPreference("gelgenie.rollingradius", 50).asObject();

    public static StringProperty deviceProperty() {
        return deviceProperty;
    }
    public static BooleanProperty useDJLProperty() {
        return useDJLProperty;
    }
    public static BooleanProperty deletePreviousBandsProperty() {
        return deletePreviousBandsProperty;
    }
    public static BooleanProperty globalCorrectionProperty() {
        return globalCorrectionProperty;
    }
    public static BooleanProperty localCorrectionProperty() {
        return localCorrectionProperty;
    }
    public static BooleanProperty rollingCorrectionProperty() {
        return rollingCorrectionProperty;
    }
    public static Property<Integer> localCorrectionPixels() {
        return localCorrectionPixels;
    }

    public static Property<Integer> rollingRadius() {
        return rollingRadius;
    }
}
