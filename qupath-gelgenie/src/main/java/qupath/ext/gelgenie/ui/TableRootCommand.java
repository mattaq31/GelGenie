package qupath.ext.gelgenie.ui;

import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.QuPathGUI;

import java.util.ResourceBundle;

import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;

import java.io.IOException;
import java.net.URL;

/**
 * Boilerplate function for generating a UI window and setting things for it to be resizable etc.
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
    private final int localSensitivity;

    public TableRootCommand(QuPathGUI qupath, String ui_name, String panel_name, Boolean resizable,
                            boolean globalCorrection, boolean localCorrection, int localSensitivity) { // Constructor
        this.qupath = qupath;
        this.panel_name = panel_name;
        this.resizable = resizable;
        URL url = getClass().getResource(ui_name + ".fxml");
        this.rootFXML = new FXMLLoader(url, resources);
        this.globalCorrection = globalCorrection;
        this.localCorrection = localCorrection;
        this.localSensitivity = localSensitivity;
    }

    @Override
    public void run() { // generates table and feeds in user settings

        // There's probably a better approach... but wrapping in a border pane
        // helped me get the resizing to behave
        BorderPane pane = null;
        try {
            pane = new BorderPane(rootFXML.load());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        stage = new Stage();
        TableController controller = rootFXML.getController();
        controller.setPreferences(globalCorrection, localCorrection, localSensitivity);
        Scene scene = new Scene(pane);

        stage.initOwner(qupath.getStage());

        stage.setTitle(resources.getString("title") + " " + panel_name);

        stage.setScene(scene);
        stage.setResizable(resizable);
        stage.show();
    }

}
