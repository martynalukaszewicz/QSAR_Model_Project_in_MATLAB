% MATLAB Script for Building a QSAR Model
% This script demonstrates the process of creating a Quantitative Structure-Activity Relationship (QSAR) model
% using synthetic chemical data. QSAR models are statistical models that relate chemical structure to biological activity.
% They are widely used in cheminformatics and drug discovery to predict how different chemical compounds will interact
% with biological targets based on their chemical features (descriptors).

% In this example, I will:
% 1. Generate synthetic data to simulate chemical descriptors and their corresponding biological activity.
% 2. Fit a linear regression model to establish a relationship between the descriptors and activity.
% 3. Evaluate the model's performance using various metrics.
% 4. Visualize the results through plots, including 2D scatter plots and a 3D surface plot.
% 5. Perform residual analysis to check the accuracy of the model.

% Ensure that you have the required toolboxes to run this script successfully.
% The script uses basic MATLAB functionalities and does not require any specialized toolboxes.

% Set the random seed for reproducibility
rng(1); % This ensures that the random numbers generated can be reproduced for consistent results

% Generate synthetic data
num_samples = 100; % Number of samples (chemicals) I will simulate
num_features = 5;  % Number of features (descriptors) for each chemical

% Generate random features (chemical descriptors)
X = rand(num_samples, num_features); 
% Here, X contains random values between 0 and 1, simulating different chemical properties

% Generate a target variable (activity) based on a linear combination of features
% Adding some noise for realism
true_coefficients = rand(num_features, 1); % Random coefficients for the linear combination
noise = randn(num_samples, 1) * 0.1; % Small Gaussian noise to make the data more realistic
y = X * true_coefficients + noise; % The target variable represents chemical activity, influenced by features

% Combine features and target variable into one matrix
chemical = [X, y]; % Combine features (X) and target variable (y) into a single dataset

% Create a table with descriptive labels for the dataset
chemical_table = array2table(chemical, 'VariableNames', ...
    {'ChemicalFeature1', 'ChemicalFeature2', 'ChemicalFeature3', 'ChemicalFeature4', 'ChemicalFeature5', 'Activity'});

% Save the synthetic dataset to a .mat file
save('chemical.mat', 'chemical_table'); % Save the dataset as a table

% Load the synthetic chemical dataset
load chemical; % Load the dataset into the workspace

% Display the first few rows of the data
disp(head(chemical_table)); % Show the first few rows to understand the structure of the dataset

% Step 1: Prepare the Data
% Extract features and target variable from the dataset
X = chemical_table{:, 1:end-1}; % All columns except the last one as features (chemical descriptors)
y = chemical_table.Activity; % Last column as the target variable (activity)

% Step 2: Build the QSAR Model
% Fit a linear regression model to predict activity based on descriptors
mdl = fitlm(X, y); % This creates a linear regression model

% Step 3: Evaluate the Model
% Display model summary to understand the relationships and statistics
disp(mdl);

% Make predictions using the fitted model on the same dataset
y_pred = predict(mdl, X);

% Calculate performance metrics (e.g., R-squared)
R2 = mdl.Rsquared.Ordinary; % R-squared value indicates how well the model explains the variability of the data
fprintf('R-squared: %.2f\n', R2); % Display the R-squared value

% Step 4: Visualize Predictions
figure; % Create a new figure window
scatter(y, y_pred, 'filled'); % Scatter plot of true activity vs predicted activity
xlabel('True Activity (Observed Values)'); % Label for the x-axis
ylabel('Predicted Activity (Model Predictions)'); % Label for the y-axis
title('Comparison of True and Predicted Activity'); % Title for the plot
axis equal; % Set equal scaling for better visual comparison
grid on; % Add grid lines for easier readability
hold on;

% Add a reference line for perfect predictions (where predicted equals true)
plot([min(y), max(y)], [min(y), max(y)], 'r--', 'LineWidth', 1.5); % Red dashed line
legend('Predicted vs True', 'Perfect Prediction Line'); % Add a legend to explain the plot
hold off;

% Save the predictions plot
saveas(gcf, 'images/qsar_model_predictions.png');

% Step 5: 3D Visualization (using the first two features)
if num_features >= 3 % Check if there are at least three features for 3D plotting
    figure; % Create a new figure for 3D visualization
    scatter3(X(:, 1), X(:, 2), y, 36, 'filled'); % 3D scatter plot for the first two features against activity
    hold on;
    % Fit a surface for better visualization using a grid of the first two features
    [X1, X2] = meshgrid(linspace(0, 1, 20), linspace(0, 1, 20)); % Create a grid for the first two features
    Y_fit = predict(mdl, [X1(:), X2(:), zeros(numel(X1), num_features-2)]); % Predict activity for the grid
    surf(X1, X2, reshape(Y_fit, size(X1)), 'FaceAlpha', 0.5, 'EdgeColor', 'none'); % Surface plot for predictions
    xlabel('Chemical Feature 1 (Descriptor 1)'); % Label for the x-axis
    ylabel('Chemical Feature 2 (Descriptor 2)'); % Label for the y-axis
    zlabel('Predicted Activity (Response)'); % Label for the z-axis
    title('3D Visualization of Activity Based on Two Features'); % Title for the plot
    grid on; % Add grid lines
    hold off;

    % Save the 3D Visualization plot
    saveas(gcf, 'images/3D_visualization_activity.png');
end

% Step 6: Residual Analysis
% Calculate residuals (the difference between observed and predicted values)
residuals = y - y_pred;

% Create a new figure for Residuals vs Predicted Activity
figure; 
scatter(y_pred, residuals, 'filled'); % Scatter plot of predicted activity vs residuals
xlabel('Predicted Activity (Model Predictions)'); % Label for the x-axis
ylabel('Residuals (True - Predicted)'); % Label for the y-axis
title('Residuals vs Predicted Activity'); % Title for the plot
grid on;

% Save the Residuals vs Predicted Activity plot
saveas(gcf, 'images/residuals_vs_predicted_activity.png');

% Create a new figure for Distribution of Residuals
figure; % Create another new figure
histogram(residuals, 20); % Histogram of residuals
xlabel('Residuals (Errors)'); % Label for the x-axis
ylabel('Frequency'); % Label for the y-axis
title('Distribution of Residuals'); % Title for the plot
grid on;

% Save the Distribution of Residuals plot
saveas(gcf, 'images/distribution_of_residuals.png');

% Step 7: Additional Performance Metrics
RMSE = sqrt(mean(residuals.^2)); % Calculate Root Mean Squared Error (RMSE) to evaluate model accuracy
fprintf('Root Mean Squared Error (RMSE): %.2f\n', RMSE); % Display the RMSE
