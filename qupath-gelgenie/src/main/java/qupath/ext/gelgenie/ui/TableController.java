package qupath.ext.gelgenie.ui;

import javafx.application.Platform;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;

import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.ImageView;
import qupath.ext.gelgenie.tools.ImageTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.regions.RegionRequest;

import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Objects;

import javafx.embed.swing.SwingFXUtils;
import qupath.lib.regions.ImageRegion;
import qupath.fx.dialogs.FileChoosers;

import static qupath.ext.gelgenie.tools.ImageTools.createAnnotationImageFrame;
import static qupath.ext.gelgenie.tools.ImageTools.extractLocalBackgroundPixels;
import static qupath.lib.scripting.QP.*;

public class TableController {

    //Table elements
    @FXML
    private TableView<BandEntry> mainTable;
    @FXML
    private TableColumn<BandEntry, ImageView> thumbnailCol;
    @FXML
    private TableColumn<BandEntry, Integer> bandCol;
    @FXML
    private TableColumn<BandEntry, String> nameCol;
    @FXML
    private TableColumn<BandEntry, Double> pixelCol;
    @FXML
    private TableColumn<BandEntry, Double> meanCol;
    @FXML
    private TableColumn<BandEntry, Double> rawCol;
    @FXML
    private TableColumn<BandEntry, Double> localVolCol;
    @FXML
    private TableColumn<BandEntry, Double> globalVolCol;
    @FXML
    private TableColumn<BandEntry, Double> normVolCol;
;

    private ObservableList<BandEntry> bandData;

    public QuPathGUI qupath;

    // user-defined settings
    private boolean globalCorrection = false;
    private boolean localCorrection = false;
    private int localSensitivity = 5;
    private final Collection<PathObject> selectedBands = new ArrayList<>();

    double globalMean = 0.0;

    public TableController() {
        this.qupath = QuPathGUI.getInstance();
    }

    @FXML
    private void initialize() {
        bandCol.setCellValueFactory(new PropertyValueFactory<>("bandID"));
        nameCol.setCellValueFactory(new PropertyValueFactory<>("bandName"));
        pixelCol.setCellValueFactory(new PropertyValueFactory<>("pixelCount"));
        meanCol.setCellValueFactory(new PropertyValueFactory<>("averageIntensity"));
        rawCol.setCellValueFactory(new PropertyValueFactory<>("rawVolume"));
        localVolCol.setCellValueFactory(new PropertyValueFactory<>("localVolume"));
        globalVolCol.setCellValueFactory(new PropertyValueFactory<>("globalVolume"));
        normVolCol.setCellValueFactory(new PropertyValueFactory<>("normVolume"));
        thumbnailCol.setCellValueFactory(new PropertyValueFactory<>("thumbnail"));

        meanCol.setCellFactory(TableController::getTableColumnTableCell);
        localVolCol.setCellFactory(TableController::getTableColumnTableCell);
        globalVolCol.setCellFactory(TableController::getTableColumnTableCell);

        ImageData<BufferedImage> imageData = getCurrentImageData();
        ImageServer<BufferedImage> server = imageData.getServer();
        Collection<PathObject> annots = getAnnotationObjects();

        // This code block depends on user settings, which are not provided until this runLater() command.
        Platform.runLater(() -> {
            calculateGlobalBackground(server);
            if (!selectedBands.isEmpty()){
                computeTableColumns(selectedBands, server);
            }
            else{
                computeTableColumns(annots, server);
            }
        });

        // permanent table settings
        mainTable.setPlaceholder(new Label("No gel band data to display"));
        TableView.TableViewSelectionModel<BandEntry> selectionModel = mainTable.getSelectionModel();
        selectionModel.setSelectionMode(SelectionMode.MULTIPLE);
    }

    private static TableCell<BandEntry, Double> getTableColumnTableCell(TableColumn<BandEntry, Double> bandEntryDoubleTableColumn) {
        return new TableCell<>() {
            @Override
            protected void updateItem(Double value, boolean empty) {
                super.updateItem(value, empty);
                if (empty) {
                    setText(null);
                } else {
                    setText(String.format("%.2f", value.floatValue()));
                }
            }
        };
    }

    /**
     * Pre-computes global background value on user-selected patch.
     * @param server: Server corresponding to open image.
     */
    private void calculateGlobalBackground(ImageServer<BufferedImage> server){
        if (globalCorrection) {
            try {
                globalMean = calculateGlobalBackgroundAverage(server);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * Computes the average pixel value of global background patches.
     * @param server: Server corresponding to current image
     * @return Computed global mean.
     * @throws Exception
     */
    private double calculateGlobalBackgroundAverage(ImageServer<BufferedImage> server) throws Exception {
        double global_mean = 0.0;
        Collection<PathObject> annots = getAnnotationObjects();
        for (PathObject annot : annots) { //TODO: what happens if multiple annotations match the correct class?
            if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Global Background")) {
                double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server);
                global_mean = Arrays.stream(all_pixels).average().getAsDouble();
            }
        }
        if (global_mean == 0.0) {
            throw new Exception("No annotation marked for global background correction");
        }
        return global_mean;
    }

    /**
     * Computes all measurements for each band in current image.
     * @param annots: List of annotations in image.
     * @param server: Server corresponding to open image.
     */
    private void computeTableColumns(Collection<PathObject> annots, ImageServer<BufferedImage> server){
        for (PathObject annot : annots) {
            //  only act on annotations marked as bands
            if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Gel Band")) {
                double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server); // extracts a list of pixels matching the specific selected annotation

                int padding = 10;  // TODO: make this user adjustable

                // Generates thumbnails for each band
                ImageRegion thumbnailRegion = createAnnotationImageFrame(annot, padding);
                RegionRequest request = RegionRequest.createInstance(server.getPath(), 1.0, thumbnailRegion);
                BufferedImage img;
                try {
                    img = server.readRegion(request);
                } catch (IOException ex) {
                    throw new RuntimeException(ex);
                }
                ImageView imviewer = new ImageView();
                imviewer.setImage(SwingFXUtils.toFXImage(img, null));

                // computes intensity volumes
                double pixel_average = Arrays.stream(all_pixels).average().getAsDouble();
                double raw_volume = Arrays.stream(all_pixels).sum();

                // todo: does it make sense to mask all bands rather than just the selected one?
                double[] localBackgroundPixels = extractLocalBackgroundPixels(annot, server, localSensitivity);
                double localMean = Arrays.stream(localBackgroundPixels).average().getAsDouble();

                double globalVolume = 0;
                double localVolume = 0;

                if (globalCorrection){
                    globalVolume = raw_volume - (globalMean * all_pixels.length);
                }
                if (localCorrection){
                    localVolume = raw_volume - (localMean * all_pixels.length);
                }

                BandEntry curr_band = new BandEntry(8, annot.getName(), all_pixels.length,
                        pixel_average, raw_volume, globalVolume, localVolume, 5.0, imviewer);

                ObservableList<BandEntry> all_bands = mainTable.getItems();
                all_bands.add(curr_band);

                bandData = all_bands;
                mainTable.setItems(all_bands);
                if (!localCorrection){ //TODO: table looks empty without these columns.  How to adjust?
                    localVolCol.setVisible(false);
                }
                if (!globalCorrection){
                    globalVolCol.setVisible(false);
                }
            }
        }
    }

    /**
     * Exports table data to CSV.
     * @throws IOException
     */
    public void exportData() throws IOException { //TODO: still rudimentary, needs updating
        File fileOutput = FileChoosers.promptToSaveFile("Export image region", null,
                FileChoosers.createExtensionFilter("Set CSV output filename", ".csv"));
        if (fileOutput == null)
            return;

        BufferedWriter br = new BufferedWriter(new FileWriter(fileOutput));

        br.write("Band Name, Pixel Count, Average Intensity, Raw Volume, Local Corrected Volume, Global Corrected Volume \n");

        for (BandEntry band : bandData) {
            String sb = band.getBandName() + "," + band.getPixelCount() + "," + band.getAverageIntensity() + "," + band.getRawVolume() + "," + band.getLocalVolume() + "," + band.getGlobalVolume() + "\n";
            br.write(sb);
        }
        br.close();
    }

    /**
     * Sets user preferences.
     * @param globalCorrection: Enable global background calculation
     * @param localCorrection: Enable local background calculation
     * @param localSensitivity: Pixel sensitivity for local background calculation
     */
    public void setPreferences(boolean globalCorrection, boolean localCorrection, int localSensitivity, Collection<PathObject> selectedBands) {
        this.localCorrection = localCorrection;
        this.globalCorrection = globalCorrection;
        this.localSensitivity = localSensitivity;
        if (!selectedBands.isEmpty()){
            this.selectedBands.addAll(selectedBands);
        }
    }

}