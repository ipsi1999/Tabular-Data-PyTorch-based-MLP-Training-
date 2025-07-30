# Predicting House Prices using a Multilayer Perceptron (MLP)

📌 Overview

This project implements a Multilayer Perceptron (MLP) using PyTorch to predict house sale prices based on various structural and area-related features. It is part of the coursework for COMPSCI 714: Deep Learning Fundamentals.

📂 Dataset

The dataset used is house_prices.csv, which contains housing data with features such as:
	•	Lot area
	•	Basement square footage
	•	Number of floors
	•	Garage size
	•	Porch and deck areas
	•	And the target variable: SalePrice

🧼 Data Preprocessing

The following preprocessing steps were carried out:
	•	Initial data inspection using pandas
	•	Selection of 14 numerical area-related features for analysis
	•	Visualization of feature distributions using histograms
	•	Correlation analysis between features and the target variable
	•	Handling of missing data using SimpleImputer
	•	Feature scaling via StandardScaler

📈 Example Visualizations

🔹 Histograms of Area Features

Histogram plots were generated for 14 surface attributes such as:
	•	LotArea, GrLivArea, GarageArea, TotalBsmtSF, etc.

These helped reveal feature distributions and outliers:

<p align="center"><img src="docs/histograms.png" alt="Histograms of Area Features" width="600"/></p>


🔹 Bar Plot of Area Bins

An area attribute with the highest correlation to SalePrice was binned and visualized:

<p align="center"><img src="docs/barplot_bins.png" alt="Binned Area Distribution" width="500"/></p>

🧠 Model Architecture

A simple feedforward Multilayer Perceptron model was built using PyTorch.
Key aspects of the architecture include:
	1. Input layer: matches number of preprocessed features
	2. Hidden layers: 3 layers (e.g., 150 → 75 → 20 neurons) with ReLU activations
	3. Output layer: single neuron (for regression)

⚙️ Training Details
	1. Loss function: Mean Squared Error (MSE)
	2. Optimizer: Adam
	3. Training/validation split: 80/20
	4. Epochs: Defined by training loop
	5. Real-time tracking of training & validation losses

📉 Loss Curve
A line plot of training and validation losses over epochs was created to show learning:

<p align="center"><img src="docs/loss_curve.png" alt="Training and Validation Losses" width="500"/></p>

🧪 Evaluation Metrics

Metric	Value
MAE (Mean Absolute Error)	~$20,000
RMSE (Root Mean Squared Error)	~$70,000

	•	The relatively high RMSE suggests some predictions were far off (outliers).
	•	Overall, the model learned well with minimal overfitting.

📊 Observations
	•	GrLivArea and GarageArea showed the strongest correlation with SalePrice.
	•	Normalization was crucial for MLP training stability.
	•	Validation loss fluctuations were reduced by tweaking learning rate & batch size.
	•	Model performance is reasonable but could benefit from:
	•	Feature selection & engineering
	•	Hyperparameter tuning
	•	Regularization techniques (e.g., Dropout, L2)
￼
Model Performance

The validation and training losses of my model consistently decreased smoothly demonstrating effective learning. As the validation loss also exhibited minor fluctuations, there might be some sensitivity to the test set. The final training and validation losses are relatively close, indicating that the model was not overfitting severely. For somepredicted samples, the predicted value is quite close to the actual value. MAE is moderate at nearly twenty thousand dollars for houses worth more than a hundred thousand dollars. RMSE is quite high at nearly seventy thousand dollars, indicating that there are some predictions far off from actual prices. 

Potential Improvements

To further enhance the model's predictive power, there are a few strategies that I would use in the future. 

1. Feature Engineering: I would select relevant features, even drop certain numerical columns that were mostly populated with ‘?’ elements.
2. Hyperparameter Tuning: I would experiment with different number of neurons per layers, learning rates, and batch sizes.  
3. Regularization: Implementing dropout layers or L2 regularization might help mitigate overfitting.
4. Better Training Dataset: I would use better data augmentation techniques could provide the model with more variability and robustness.

Challenges Faced & Solutions

One challenge was ensuring stability in validation loss, as it fluctuated slightly across epochs. This was addressed by adjusting the learning rate and batch size. Another issue was the initial high loss, which was mitigated by fine-tuning weight initialization and optimizer settings. Additionally, missing values in the dataset were handled appropriately to avoid errors during training.

Overall, I think that the model performed reasonably well but would benefit from further optimization techniques to improve accuracy and generalization.
