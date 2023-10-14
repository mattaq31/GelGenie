package qupath.ext.gelgenie.graphics;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.Scene;
import javafx.scene.chart.*;
import javafx.stage.Stage;

import java.util.Arrays;


// TODO: remove this file if unnecessary.
public class GelGenieBarChart{

    public void plot(double[] y_data, int bins) {

        Stage stage = new Stage();
        stage.setTitle("Bar Chart Sample");
        final CategoryAxis xAxis = new CategoryAxis();
        final NumberAxis yAxis = new NumberAxis();
        final BarChart<String,Number> bc = new BarChart<String,Number>(xAxis,yAxis);

        bc.setTitle("Selected Annotation Pixel Histogram");
        xAxis.setLabel("Pixel Intensity");
        yAxis.setLabel("Frequency");
        bc.setBarGap(0);
        bc.setCategoryGap(0);

        // max/min vals
        double max_val = Arrays.stream(y_data).max().getAsDouble();
        double min_val = Arrays.stream(y_data).min().getAsDouble();
        double mean = Arrays.stream(y_data).average().getAsDouble();

        // Calculate the bin width
        double binWidth = (max_val - min_val) / bins;

        int meanIndex = (int) ((mean - min_val) / binWidth);
        double meanBinStart = min_val + meanIndex * binWidth;
        double meanBinEnd = min_val + (meanIndex + 1) * binWidth;

        // Create 10 bins and initialize their counts to 0
        int[] bin_values = new int[bins];

        for (double value: y_data) {
            int binIndex = (int) ((value - min_val) / binWidth);
            if (binIndex < 0) {
                binIndex = 0;
            } else if (binIndex >= bins) {
                binIndex = bins-1;
            }
            bin_values[binIndex]++;
        }

        ObservableList<XYChart.Data<String, Number>> list = FXCollections.observableArrayList();
        // Add all values from the array into the list
        for (int i = 0; i < bin_values.length; i++) {
            double binStart = min_val + i * binWidth;
            double binEnd = min_val + (i + 1) * binWidth;
            list.add(new XYChart.Data(String.format("%.1f - %.1f", binStart, binEnd), bin_values[i]));
        }
        // create a series from the list
        XYChart.Series<String, Number> stst = new XYChart.Series<String, Number>(list);
        stst.setName("Histogram");
        // Add the series
        bc.getData().addAll(stst);

        Scene scene  = new Scene(bc,800,600);

        // Set custom colors for the bars
        for (XYChart.Data<String, Number> data : stst.getData()) {
            // Change the color of each bar to blue
            data.getNode().setStyle("-fx-bar-fill: #002aff;");
        }
        bc.setLegendVisible(false);

        // Create a vertical line chart to plot the mean
        CategoryAxis meanXAxis = new CategoryAxis();
        NumberAxis meanYAxis = new NumberAxis();
        LineChart<String, Number> meanLineChart = new LineChart<>(meanXAxis, meanYAxis);
        meanLineChart.setLegendVisible(false);

        XYChart.Series<String, Number> meanSeries = new XYChart.Series<>();
        meanSeries.getData().add(new XYChart.Data<>(String.format("Mean (%.2f)", mean), 0));
        meanSeries.getData().add(new XYChart.Data<>(String.format("Mean (%.2f)", mean), 1));
        // Add the mean line chart to the scene
//        scene.getRoot().getChildrenUnmodifiable().add(meanLineChart);

        stage.setScene(scene);
        stage.show();
    }

}
