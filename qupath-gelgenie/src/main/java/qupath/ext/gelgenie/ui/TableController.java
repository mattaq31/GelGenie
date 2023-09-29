package qupath.ext.gelgenie.ui;

import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.control.SelectionMode;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.cell.PropertyValueFactory;
import qupath.lib.objects.PathObject;

import java.util.Collection;

import static qupath.lib.scripting.QP.getAnnotationObjects;

public class TableController {

    //Table
    @FXML
    private TableView<BandEntry> mainTable;

    @FXML
    private TableColumn<BandEntry, String> bandCol;

    @FXML
    private TableColumn<BandEntry, Integer> areaCol;

    public TableController(){
        // set selection mode to multiple rows
    }

    @FXML
    private void initialize() {
        bandCol.setCellValueFactory(new PropertyValueFactory<BandEntry, String>("bandID"));
        areaCol.setCellValueFactory(new PropertyValueFactory<BandEntry, Integer>("area"));

        Collection<PathObject> annots = getAnnotationObjects();
        for (PathObject annot : annots){
            BandEntry curr_band = new BandEntry(annot.getPathClass().toString(), (int) annot.getROI().getArea());
            ObservableList<BandEntry> all_bands = mainTable.getItems();
            all_bands.add(curr_band);
            mainTable.setItems(all_bands);
        }
        mainTable.setPlaceholder(new Label("No gel band data to display"));
        TableView.TableViewSelectionModel<BandEntry> selectionModel = mainTable.getSelectionModel();
        selectionModel.setSelectionMode(SelectionMode.MULTIPLE);
        bandCol.prefWidthProperty().bind(mainTable.widthProperty().multiply(0.3));
        areaCol.prefWidthProperty().bind(mainTable.widthProperty().multiply(0.7));


//        BandEntry test_table = new BandEntry("TEST",5);

//        ObservableList<BandEntry> all_bands = mainTable.getItems();
//        all_bands.add(test_table);
//        mainTable.setItems(all_bands);
        }
}
