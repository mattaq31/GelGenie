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

package qupath.ext.gelgenie;

import javafx.beans.property.BooleanProperty;
import javafx.scene.control.MenuItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.prefs.PathPrefs;

import java.util.*;

import qupath.ext.gelgenie.ui.GUIRootCommand;

/**
 This is the main access point for all the GelGenie functionality.
 */
public class GelGenie implements QuPathExtension, GitHubProject {

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

    @Override
    public GitHubRepo getRepository() {
        return GitHubRepo.create(getName(), "mattaq31", "GelGenie");
    }
}
