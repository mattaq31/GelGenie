package qupath.ext.gelgenie.ui;

import javafx.fxml.FXML;
import javafx.scene.control.CheckBox;
import qupath.lib.gui.QuPathGUI;

import java.io.IOException;

public class TablePreferences {

    public QuPathGUI qupath;
    @FXML
    private CheckBox bandCheckBox;
    @FXML
    private CheckBox laneCheckBox;
    @FXML
    private CheckBox nameCheckBox;
    @FXML
    private CheckBox pixelsCheckBox;
    @FXML
    private CheckBox widthCheckBox;
    @FXML
    private CheckBox heightCheckBox;
    @FXML
    private CheckBox averageCheckBox;
    @FXML
    private CheckBox sdCheckBox;
    @FXML
    private CheckBox rawCheckBox;
    @FXML
    private CheckBox normRawCheckBox;
    @FXML
    private CheckBox localCheckBox;
    @FXML
    private CheckBox normLocalCheckBox;
    @FXML
    private CheckBox globalCheckBox;
    @FXML
    private CheckBox normGlobalCheckBox;
    @FXML
    private CheckBox rollingCheckBox;
    @FXML
    private CheckBox normRollingCheckBox;

    public TablePreferences() {
        this.qupath = QuPathGUI.getInstance();
    }

    @FXML
    private void initialize() throws IOException {
        // checkboxes directly modify the persistent GelGenie properties, which also immediately update the data table
        this.bandCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.band"));
        this.laneCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.lane"));
        this.nameCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.name"));
        this.pixelsCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.pixelcount"));
        this.widthCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.width"));
        this.heightCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.height"));
        this.averageCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.averagepixel"));
        this.sdCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.sdpixel"));
        this.rawCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.rawvol"));
        this.normRawCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.normrawvol"));
        this.localCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.localvol"));
        this.normLocalCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.normlocalvol"));
        this.globalCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.globalvol"));
        this.normGlobalCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.normglobalvol"));
        this.rollingCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.rbvol"));
        this.normRollingCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.specificDataBoolPref("gelgenie.data.normrbvol"));
    }
}
