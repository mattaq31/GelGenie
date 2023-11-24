package qupath.ext.gelgenie.ui;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.StringProperty;
import qupath.lib.gui.prefs.PathPrefs;

public class GelGeniePrefs {
    private static final StringProperty deviceProperty = PathPrefs.createPersistentPreference("gelgenie.device", "cpu");
    private static final BooleanProperty useDJLProperty = PathPrefs.createPersistentPreference("gelgenie.djl", false);
    public static StringProperty deviceProperty() {
        return deviceProperty;
    }
    public static BooleanProperty useDJLProperty() {
        return useDJLProperty;
    }
}
