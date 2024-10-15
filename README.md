# QSAR Model in MATLAB

## Overview
This MATLAB script demonstrates the creation of a Quantitative Structure-Activity Relationship (QSAR) model using synthetic chemical data. QSAR models are statistical models that relate chemical structure to biological activity, and they are widely used in cheminformatics and drug discovery.

## Objectives
In this example, I will:
1. Generate synthetic data (chemical.mat) to simulate chemical descriptors and their corresponding biological activity.
2. Fit a linear regression model to establish a relationship between the descriptors and activity.
3. Evaluate the model's performance using various metrics.
4. Visualize the results through plots, including 2D scatter plots and a 3D surface plot.
5. Perform residual analysis to check the accuracy of the model.

## Requirements
- MATLAB with basic functionalities.
- Ensure the script runs without any specialized toolboxes.

## Generated Graphics
The script produces the following plots:

### 1. Comparison of True and Predicted Activity
![Comparison of True and Predicted Activity](images/comparison_plot.png)

### 2. 3D Visualization of Activity Based on Two Features
![3D Visualization of Activity](https://raw.githubusercontent.com/martynalukaszewicz/QSAR_Model_Project_in_MATLAB/main/images/3D_visualization_activity.png)

### 3. Residuals vs Predicted Activity
![Residuals vs Predicted Activity](images/residuals_plot.png)

### 4. Distribution of Residuals
![Distribution of Residuals](images/residuals_histogram.png)

## How to Run the Script
1. Ensure you have MATLAB installed.
2. Save the MATLAB script as `qsar_model.m`.
3. Run the script in MATLAB. The synthetic dataset will be generated, and various plots will be displayed.


