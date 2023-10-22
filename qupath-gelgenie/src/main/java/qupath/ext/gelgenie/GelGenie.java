package qupath.ext.gelgenie;

import javafx.beans.property.BooleanProperty;
import javafx.scene.control.MenuItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.prefs.PathPrefs;

import java.util.*;

import qupath.ext.gelgenie.ui.GUIRootCommand;

/**
 This is the main access point for all the GelGenie functionality.
 */
public class GelGenie implements QuPathExtension {

    private final static ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");
    private final static Logger logger = LoggerFactory.getLogger(QuPathExtension.class);
    private final static String EXTENSION_NAME = resources.getString("extension.title");
    private final static String EXTENSION_DESCRIPTION = resources.getString("extension.description");
    private final static Version EXTENSION_QUPATH_VERSION = Version.parse(resources.getString("extension.qupath.version"));

    private boolean isInstalled = false;

    private BooleanProperty enableExtensionProperty = PathPrefs.createPersistentPreference(
            "enableExtension", true);

    @Override
    public void installExtension(QuPathGUI qupath) {
        if (isInstalled) {
            logger.debug("{} is already installed", getName());
            return;
        }
        isInstalled = true;
        addPreference(qupath);
        addMenuItems(qupath);
    }

    /**
     Users can disable the extension from the preferences pane rather than deleting the whole thing.
     */
    private void addPreference(QuPathGUI qupath) {
        qupath.getPreferencePane().addPropertyPreference(
                enableExtensionProperty,
                Boolean.class,
                "Enable extension",
                EXTENSION_NAME,
                "Enable extension");
    }

    /**
     * Main access point to extension GUI is from a menu item.  This function sets everything up.
     */
    private void addMenuItems(QuPathGUI qupath) {
        var menu = qupath.getMenu("Extensions>" + EXTENSION_NAME, true);
        MenuItem menuItem = new MenuItem("Activate GelGenie");

        GUIRootCommand command = new GUIRootCommand(qupath, "gelgenie_control", "None", false); // activation class from ui folder

        menuItem.setOnAction(e -> {
            command.run();
        });
        menuItem.disableProperty().bind(enableExtensionProperty.not());
        menu.getItems().add(menuItem);

    }

    @Override
    public String getName() {
        return EXTENSION_NAME;
    }

    @Override
    public String getDescription() {
        return EXTENSION_DESCRIPTION;
    }

    @Override
    public Version getQuPathVersion() {
        return EXTENSION_QUPATH_VERSION;
    }

}
