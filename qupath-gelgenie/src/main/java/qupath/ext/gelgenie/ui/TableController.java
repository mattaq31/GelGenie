package qupath.ext.gelgenie.ui;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.collections.ListChangeListener;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;

import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.gelgenie.graphics.EmbeddedBarChart;
import qupath.ext.gelgenie.tools.ImageTools;
import qupath.ext.gelgenie.tools.laneBandCompare;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;

import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.*;
import java.util.concurrent.ExecutorService;

import qupath.fx.dialogs.FileChoosers;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.objects.hierarchy.events.PathObjectSelectionModel;

import static qupath.ext.gelgenie.graphics.EmbeddedBarChart.saveChart;
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
    private TableColumn<BandEntry, Double> normLocalVolCol;
    @FXML
    private TableColumn<BandEntry, Double> normGlobalVolCol;
    @FXML
    private SplitPane dataTableSplitPane;

    private ObservableList<BandEntry> bandData;
    private Boolean barChartActive = false;
    private BarChart<String, Number> displayChart;

    public QuPathGUI qupath;
    private static final Logger logger = LoggerFactory.getLogger(TableController.class);

    // user-defined settings
    private boolean globalCorrection = false;
    private boolean localCorrection = false;
    private int localSensitivity = 5;
    private final Collection<PathObject> selectedBands = new ArrayList<>();

    double globalMean = 0.0;
    private final static ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");


    public TableController() {
        this.qupath = QuPathGUI.getInstance();
    }

    @FXML
    private void initialize() {
        // links table columns to properties of BandEntry class
        bandCol.setCellValueFactory(new PropertyValueFactory<>("bandID"));
        laneCol.setCellValueFactory(new PropertyValueFactory<>("laneID"));
        nameCol.setCellValueFactory(new PropertyValueFactory<>("bandName"));
        pixelCol.setCellValueFactory(new PropertyValueFactory<>("pixelCount"));
        meanCol.setCellValueFactory(new PropertyValueFactory<>("averageIntensity"));
        rawCol.setCellValueFactory(new PropertyValueFactory<>("rawVolume"));
        localVolCol.setCellValueFactory(new PropertyValueFactory<>("localVolume"));
        globalVolCol.setCellValueFactory(new PropertyValueFactory<>("globalVolume"));
        normVolCol.setCellValueFactory(new PropertyValueFactory<>("normVolume"));
        normLocalVolCol.setCellValueFactory(new PropertyValueFactory<>("normLocal"));
        normGlobalVolCol.setCellValueFactory(new PropertyValueFactory<>("normGlobal"));
        thumbnailCol.setCellValueFactory(new PropertyValueFactory<>("parentAnnotation"));

        // formatting for double columns
        meanCol.setCellFactory(TableController::getTableFormattedDouble);
        localVolCol.setCellFactory(TableController::getTableFormattedDouble);
        globalVolCol.setCellFactory(TableController::getTableFormattedDouble);
        normVolCol.setCellFactory(TableController::getTableFormattedDouble);
        normLocalVolCol.setCellFactory(TableController::getTableFormattedDouble);
        normGlobalVolCol.setCellFactory(TableController::getTableFormattedDouble);

        // getting data from viewer
        ImageData<BufferedImage> imageData = getCurrentImageData();
        ImageServer<BufferedImage> server = imageData.getServer();
        final PathObjectHierarchy hierarchy = imageData.getHierarchy();
        var viewer = qupath.getAllViewers().stream().filter(v -> v.getImageData() == imageData).findFirst().orElse(null);

        // uses the implementation in qupath to extract a thumbnail from an annotation
        thumbnailCol.setCellFactory(column -> createTableCellByReflection(
                viewer, imageData.getServer(), true, 5,
                qupath.getThreadPoolManager().getSingleThreadExecutor(this)));

        // Set fixed cell size for the thumbnail - this can avoid large numbers of non-visible cells being computed
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

        // the below synchronises selections on the table with selections in QuPath (and the histogram viewer)
        // However, the opposite is currently not implemented (would require more code from SummaryMeasurementTableCommand).
        mainTable.getSelectionModel().getSelectedItems().addListener(new ListChangeListener<>() {
            @Override
            public void onChanged(ListChangeListener.Change<? extends BandEntry> c) {
                synchronizeSelectionModelToTable(hierarchy, c, mainTable);
            }
        });

    }

    /**
     * Formats double columns to two decimal places.
     * @param bandEntryDoubleTableColumn Column being edited.
     * @return Edited value.
     */
    private static TableCell<BandEntry, Double> getTableFormattedDouble(TableColumn<BandEntry, Double> bandEntryDoubleTableColumn) {
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
                globalCorrection = false;
                Dialogs.showInfoNotification(
                        resources.getString("title"),
                        resources.getString("ui.processing.no-global-mean"));
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
        for (PathObject annot : annots) {
            if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Global Background")) {
                double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server);
                global_mean = global_mean + Arrays.stream(all_pixels).average().getAsDouble();
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

        double inf = Double.POSITIVE_INFINITY;
        double[] minMax = {inf,-inf};
        double[] globalMinMax = {inf,-inf};
        double[] localMinMax = {inf,-inf};

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
                        pixel_average, raw_volume, globalVolume, localVolume, annot);

                ObservableList<BandEntry> all_bands = mainTable.getItems();
                all_bands.add(curr_band);

                bandData = all_bands;
                mainTable.setItems(all_bands);

                // updates min/max values for normalization
                minMax[0] = Math.min(minMax[0], raw_volume);
                minMax[1] = Math.max(minMax[1], raw_volume);
                localMinMax[0] = Math.min(localMinMax[0], localVolume);
                localMinMax[1] = Math.max(localMinMax[1], localVolume);
                globalMinMax[0] = Math.min(globalMinMax[0], globalVolume);
                globalMinMax[1] = Math.max(globalMinMax[1], globalVolume);
            }
        }

        // normalised values can only be updated when everything else is complete
        ObservableList<BandEntry> all_bands = mainTable.getItems();
        for (BandEntry entry:all_bands){
            entry.setNormVolume((entry.getRawVolume() - minMax[0]) / (minMax[1] - minMax[0]));
            entry.setNormLocal((entry.getLocalVolume() - localMinMax[0]) / (localMinMax[1] - localMinMax[0]));
            entry.setNormGlobal((entry.getGlobalVolume() - globalMinMax[0]) / (globalMinMax[1] - globalMinMax[0]));
        }

        if (!localCorrection){ //TODO: table looks empty without these columns.  How to adjust?
            localVolCol.setVisible(false);
            normLocalVolCol.setVisible(false);
        }
        if (!globalCorrection){
            globalVolCol.setVisible(false);
            normGlobalVolCol.setVisible(false);
        }
    }

    /**
     * Exports salient table columns to CSV.
     * @throws IOException
     */
    public void exportData() throws IOException {
        File fileOutput = FileChoosers.promptToSaveFile("Export image region", new File("bandData.csv"),
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

    /**
     * Toggles the display of a histogram, which appears as a side panel in the table view.
     */
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

    /**
     * Collects data from table and prepares histogram.  Currently only presents raw and global corrected volumes.
     */
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

    /**
     * Saves histogram to file.
     */
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


    /**
     * Provides access to QuPath's tablecell thumbnail creation method via reflection (provided by Pete).
     * @param viewer: Current image viewer
     * @param server: Server corresponding to current image
     * @param paintObject: Set to true to paint out the annotation within the thumbnail
     * @param padding: Padding to include around image
     * @param pool: Thread pool
     * @return Updated table cell with thumbnail
     */
    public static <S extends PathObject, T extends PathObject> TableCell<S, T>  createTableCellByReflection(
            QuPathViewer viewer, ImageServer<BufferedImage> server, boolean paintObject, double padding, ExecutorService pool
    ) {
        Class<?> cls = null;
        try {
            cls = Class.forName("qupath.lib.gui.commands.PathObjectImageManagers");
            var method = cls.getDeclaredMethod("createTableCell",
                    QuPathViewer.class,
                    ImageServer.class,
                    boolean.class,
                    double.class,
                    ExecutorService.class
            );
            method.setAccessible(true);
            return (TableCell)method.invoke(null, viewer, server, paintObject, padding, pool);
        } catch (ClassNotFoundException | InvocationTargetException | NoSuchMethodException | IllegalAccessException e) {
            logger.warn("Exception creating table cell: {}", e.getMessage(), e);
            return new TableCell<>();
        }
    }

    // The below was modified from QuPath's Summary MeasurementTableCommand
    private boolean synchronizingTableToModel = false;
    private boolean synchronizingModelToTable = false;
    private void synchronizeSelectionModelToTable(final PathObjectHierarchy hierarchy, final ListChangeListener.Change<? extends BandEntry> change, final TableView<BandEntry> table) {
        if (synchronizingTableToModel || hierarchy == null)
            return;

        PathObjectSelectionModel model = hierarchy.getSelectionModel();
        if (model == null) {
            return;
        }

        boolean wasSynchronizingToTree = synchronizingModelToTable;
        try {
            synchronizingModelToTable = true;

            // Check - was anything removed?
            boolean removed = false;
            if (change != null) {
                while (change.next())
                    removed = removed | change.wasRemoved();
            }

            MultipleSelectionModel<BandEntry> treeModel = table.getSelectionModel();
            List<BandEntry> selectedItems = treeModel.getSelectedItems();

            // If we just have no selected items, and something was removed, then clear the selection
            if (selectedItems.isEmpty() && removed) {
                model.clearSelection();
                return;
            }

            // If we just have one selected item, and also items were removed from the selection, then only select the one item we have
//			if (selectedItems.size() == 1 && removed) {
            if (selectedItems.size() == 1) {
                model.setSelectedObject(selectedItems.get(0).getParentAnnotation(), false);
                return;
            }

            // If we have multiple selected items, we need to ensure that everything in the tree matches with everything in the selection model
            Set<BandEntry> toSelect = new HashSet<>(treeModel.getSelectedItems());
            Collection<PathObject> annotSelect = new ArrayList<PathObject>();

            for(BandEntry entry : toSelect) {
            	annotSelect.add(entry.getParentAnnotation());
            }

            PathObject primary = treeModel.getSelectedItem().getParentAnnotation();
            model.setSelectedObjects(annotSelect, primary);
        } finally {
            synchronizingModelToTable = wasSynchronizingToTree;
        }
    }

}