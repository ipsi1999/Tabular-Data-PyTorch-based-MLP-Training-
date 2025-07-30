# Tabular-Data-PyTorch-based-MLP-Training-
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
