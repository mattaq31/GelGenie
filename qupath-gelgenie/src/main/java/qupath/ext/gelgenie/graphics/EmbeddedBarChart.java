package qupath.ext.gelgenie.graphics;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.chart.*;

import java.util.Arrays;

public class EmbeddedBarChart {

    public XYChart.Series<String, Number> plot(double[] y_data, int bins) {

        // max/min vals
        double max_val = Arrays.stream(y_data).max().getAsDouble();
        double min_val = Arrays.stream(y_data).min().getAsDouble();
        double mean = Arrays.stream(y_data).average().getAsDouble();

        // Calculate the bin width
        double binWidth = (max_val - min_val) / bins;

        int meanIndex = (int) ((mean - min_val) / binWidth);

        // Create bins and initialize their counts to 0
        int[] bin_values = new int[bins];

        for (double value: y_data) { // collects count data
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
        XYChart.Series<String, Number> stst = new XYChart.Series<>(list);
        stst.setName("Histogram");

        return stst;

    }

}
