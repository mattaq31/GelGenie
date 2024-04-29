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

package qupath.ext.gelgenie.ui;

import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.ExtensionClassLoader;
import qupath.lib.gui.QuPathGUI;

import java.util.ArrayList;
import java.util.Collection;
import java.util.ResourceBundle;

import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import qupath.lib.objects.PathObject;

import java.io.IOException;
import java.net.URL;

/**
 * Creates the table window and properly sets things up for it to be resizable.
 */
public class TableRootCommand implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(GUIRootCommand.class);
    private final QuPathGUI qupath;
    private final String panel_name;
    private final Boolean resizable;
    private final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");
    private Stage stage;
    private final FXMLLoader rootFXML;

    private final boolean globalCorrection;
    private final boolean localCorrection;
    private final boolean rollingCorrection;
    private final int localSensitivity;
    private final int rollingRadius;
    private final boolean invertImage;

    private final Collection<PathObject> selectedBands = new ArrayList<>();

    public TableRootCommand(QuPathGUI qupath, String ui_name, String panel_name, Boolean resizable,
                            boolean globalCorrection, boolean localCorrection, boolean rollingCorrection,
                            int localSensitivity, int rollingRadius, boolean invertImage,
                            Collection<PathObject> selectedBands) { // Constructor
        this.qupath = qupath;
        this.panel_name = panel_name;
        this.resizable = resizable;
        URL url = getClass().getResource(ui_name + ".fxml");
        this.rootFXML = new FXMLLoader(url, resources);
        rootFXML.setClassLoader(ExtensionClassLoader.getInstance());

        // user defined settings
        this.globalCorrection = globalCorrection;
        this.localCorrection = localCorrection;
        this.rollingCorrection = rollingCorrection;
        this.localSensitivity = localSensitivity;
        this.rollingRadius = rollingRadius;
        this.invertImage = invertImage;

        if (!selectedBands.isEmpty()){
            this.selectedBands.addAll(selectedBands);
        }
    }

    /**
     * Generates table window and feeds in user settings.
     */
    @Override
    public void run() {

        // There's probably a better approach... but wrapping in a border pane
        // helped me get the resizing to behave TODO: is this necessary anymore?
        BorderPane pane;
        try {
            pane = new BorderPane(rootFXML.load());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        stage = new Stage();
        TableController controller = rootFXML.getController();
        controller.setPreferences(globalCorrection, localCorrection, rollingCorrection, localSensitivity,
                rollingRadius, invertImage, selectedBands);

        Scene scene = new Scene(pane);
        stage.initOwner(qupath.getStage());
        stage.setTitle(resources.getString("title") + " " + panel_name);

        stage.setScene(scene);
        stage.setResizable(resizable);
        stage.show();
    }

}
