Manuscript Revisions Code
==============================
- This folder contains additional code that was used to add figures/tables to the original gelgenie paper post peer review.  In  brief:
  - To generate the standard set regression results, run the code in `image_studio_figure_1.ipynb`. You will need to have the gelanalyzer, gelgenie and image studio results prepared to be able to run this notebook.
  - To generate the model results on the new external 25-image dataset, first run `external_set_eval.py`.  You will need to point the filepaths to where your saved models and dataset are stored.  Keep in mind that two runs are required - one for standard normalization and one for percentile normalization.
  - Next, run `band_level_statistics_test_set_figure_3.py` to obtain band-level accuracy statistics.  You will need to run this twice, once for each normalization method.
  - Then, use `figure_3_graphs_and_tables.ipynb` to generate all the figures and table data for the model supplementary results.
  - `figure_3_full_metric_test_set_eval.py` is used to re-run the test set analysis for the original figure 3.  This now includes additional metrics: precision, recall and Hausdorff distance.
  - `presenting_external_test_set_segmentation_maps.ipynb` is used to create the supplementary figures comparing the segmentation maps for the external test set.
  - `dataset_summary.ipynb` is used to generate the figures comparing the intensities of the different datasets.

ImageStudio and GelAnalyzer data is available for download from our Zenodo deposition [here](https://doi.org/10.5281/zenodo.14827469).
