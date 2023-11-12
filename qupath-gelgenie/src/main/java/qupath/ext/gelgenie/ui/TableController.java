package qupath.ext.gelgenie.ui;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;

import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.ImageView;
import qupath.ext.gelgenie.graphics.EmbeddedBarChart;
import qupath.ext.gelgenie.tools.ImageTools;
import qupath.ext.gelgenie.tools.laneBandCompare;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.images.servers.RenderedImageServer;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.regions.RegionRequest;

import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import javafx.embed.swing.SwingFXUtils;
import qupath.lib.regions.ImageRegion;
import qupath.fx.dialogs.FileChoosers;

import static qupath.ext.gelgenie.graphics.EmbeddedBarChart.saveChart;
import static qupath.ext.gelgenie.tools.ImageTools.createAnnotationImageFrame;
import static qupath.ext.gelgenie.tools.ImageTools.extractLocalBackgroundPixels;
import static qupath.lib.scripting.QP.*;

public class TableController {

    //Table elements
    @FXML
    private TableView<BandEntry> mainTable;
    @FXML
    private TableColumn<BandEntry, PathObject> thumbnailCol;
    @FXML
    private TableColumn<BandEntry, Integer> bandCol;
    @FXML
    private TableColumn<BandEntry, Integer> laneCol;
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
    @FXML
    private SplitPane dataTableSplitPane;

    private ObservableList<BandEntry> bandData;
    private Boolean barChartActive = false;
    private BarChart<String, Number> displayChart;

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
        laneCol.setCellValueFactory(new PropertyValueFactory<>("laneID"));
        nameCol.setCellValueFactory(new PropertyValueFactory<>("bandName"));
        pixelCol.setCellValueFactory(new PropertyValueFactory<>("pixelCount"));
        meanCol.setCellValueFactory(new PropertyValueFactory<>("averageIntensity"));
        rawCol.setCellValueFactory(new PropertyValueFactory<>("rawVolume"));
        localVolCol.setCellValueFactory(new PropertyValueFactory<>("localVolume"));
        globalVolCol.setCellValueFactory(new PropertyValueFactory<>("globalVolume"));
        normVolCol.setCellValueFactory(new PropertyValueFactory<>("normVolume"));
        thumbnailCol.setCellValueFactory(new PropertyValueFactory<>("parentAnnotation"));

        meanCol.setCellFactory(TableController::getTableColumnTableCell);
        localVolCol.setCellFactory(TableController::getTableColumnTableCell);
        globalVolCol.setCellFactory(TableController::getTableColumnTableCell);


        ImageData<BufferedImage> imageData = getCurrentImageData();
        ImageServer<BufferedImage> server = imageData.getServer();
        var viewer = qupath.getAllViewers().stream().filter(v -> v.getImageData() == imageData).findFirst().orElse(null);

        // uses the implementation in qupath to extract a thumbnail from an annotation
        thumbnailCol.setCellFactory(column -> thumbnailManager.createTableCell(
                viewer, imageData.getServer(), true, 5,
                qupath.getThreadPoolManager().getSingleThreadExecutor(this)));

        // Set fixed cell size - this can avoid large numbers of non-visible cells being computed
        mainTable.fixedCellSizeProperty().bind(Bindings.createDoubleBinding(() -> {
            if (thumbnailCol.isVisible())
                return Math.max(24, thumbnailCol.getWidth() + 5);
            else
                return -1.0;
        }, thumbnailCol.widthProperty(), thumbnailCol.visibleProperty()));

        ArrayList<PathObject> annots = (ArrayList<PathObject>) getAnnotationObjects();

        // This code block depends on user settings, which are not provided until this runLater() command.
        Platform.runLater(() -> {
            calculateGlobalBackground(server);
            if (!selectedBands.isEmpty()){
                computeTableColumns(selectedBands, server);
            }
            else{
                annots.sort(new laneBandCompare());
                computeTableColumns(annots, server);
            }
            // toggleHistogram(); Having the histogram pop up by default can be annoying sometimes.
        });

        // permanent chart settings
        final CategoryAxis xAxis = new CategoryAxis();
        final NumberAxis yAxis = new NumberAxis();
        displayChart = new BarChart<>(xAxis, yAxis);
        displayChart.setTitle("Visual Data Depiction");
        displayChart.lookup(".chart-plot-background").setStyle("-fx-background-color: transparent;");
        displayChart.lookup(".chart-legend").setStyle("-fx-background-color: transparent;");
        xAxis.setLabel("Band");
        yAxis.setLabel("Quantity");

        // permanent table settings
        mainTable.setPlaceholder(new Label("No gel band data to display"));
        TableView.TableViewSelectionModel<BandEntry> selectionModel = mainTable.getSelectionModel();
        selectionModel.setSelectionMode(SelectionMode.MULTIPLE);

        // when the table is double-clicked, the viewer zooms in on the selected band
        mainTable.setRowFactory(params -> {
            var row = new TableRow<BandEntry>();
            row.setOnMouseClicked(e -> {
                if (e.getClickCount() == 2) {
                    if (row.getItem() == null)
                        return;
                    var roi = row.getItem().getParentAnnotation().getROI();
                    if (roi != null && viewer != null && viewer.getHierarchy() != null)
                        viewer.centerROI(roi);
                }
            });
            return row;
        });

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

                int bandID;
                int laneID;
                if (annot.getMeasurements().get("BandID") == null){
                    bandID = 0;
                }
                else{
                    bandID = annot.getMeasurements().get("BandID").intValue();
                }

                if (annot.getMeasurements().get("LaneID") == null){
                    laneID = 0;
                }
                else{
                    laneID = annot.getMeasurements().get("LaneID").intValue();
                }

                BandEntry curr_band = new BandEntry(bandID, laneID, annot.getName(), all_pixels.length,
                        pixel_average, raw_volume, globalVolume, localVolume, 5.0, annot);

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

        br.write("Lane ID, Band ID, Pixel Count, Average Intensity, Raw Volume, Local Corrected Volume, Global Corrected Volume \n");

        for (BandEntry band : bandData) {
            String sb = band.getLaneID() + "," + band.getBandID() + "," + band.getPixelCount() + "," + band.getAverageIntensity() + "," + band.getRawVolume() + "," + band.getLocalVolume() + "," + band.getGlobalVolume() + "\n";
            br.write(sb);
        }
        br.close();
    }

    public void toggleHistogram(){
        if (barChartActive){
            dataTableSplitPane.getItems().remove(1);
            barChartActive = false;
        }
        else {
            updateHistogramData();
            dataTableSplitPane.getItems().add(displayChart);
            barChartActive = true;
        }
    }

    private void updateHistogramData(){

        Collection<double[]> dataList = new ArrayList<>();
        Collection<String> legendList = new ArrayList<>();

        ObservableList<BandEntry> all_bands = mainTable.getItems();
        double[] rawPixels = new double[all_bands.size()];
        double[] globalCorrVol = new double[all_bands.size()];

        String[] labels = new String[all_bands.size()];
        int counter = 0;

        for (BandEntry band : all_bands) {
            rawPixels[counter] = band.getRawVolume();
            labels[counter] = band.getBandName();
            if (this.globalCorrection){
                globalCorrVol[counter] = band.getGlobalVolume();
            }
            counter++;
        }
        dataList.add(rawPixels);
        legendList.add("Raw Volume");

        if (this.globalCorrection){
            dataList.add(globalCorrVol);
            legendList.add("Global Corrected Volume");
        }
        ObservableList<XYChart.Series<String, Number>> allPlots = EmbeddedBarChart.plotBars(dataList, legendList, labels);

        displayChart.getData().clear(); // removes previous data
        displayChart.getData().addAll(allPlots);
    }

    public void saveHistogram(){
        saveChart(displayChart);
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