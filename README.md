# Mobile-range-Prediction-using-Random-Forest
This project aims to predict the price range of mobile phones based on their specifications using a Random Forest classifier. The dataset, publicly available, includes features such as battery power, Bluetooth connectivity, clock speed, and more. The project involves several key steps: data exploration, preprocessing, feature engineering, model training, evaluation, tuning, and deployment.

Data preprocessing ensures the dataset is clean, while feature engineering selects the most relevant features for prediction. The Random Forest model is trained on this refined dataset and evaluated using metrics like accuracy, precision, and recall. A confusion matrix is also generated to visualize model performance.

Hyperparameter tuning is performed through cross-validation to optimize the model. The trained model is then used to predict the price range of a mobile phone based on its specifications. The project concludes with a discussion on deploying the model as a web application using frameworks like Flask or Streamlit.

The repository includes the Python script for the entire process, a saved model, and a detailed report in a Word document. This project demonstrates an effective approach to mobile price prediction using machine learning.



# Data Exploration



![download](https://github.com/user-attachments/assets/90d0f565-2b34-46e6-8ae8-509e892905ee)



# Visualization of Key Features

![Figure_2](https://github.com/user-attachments/assets/8de5eb91-37b5-4a55-982f-29769d186e50)
![Figure_3](https://github.com/user-attachments/assets/4e705fa4-e89b-4a5d-9014-7d7b94de8875)
![Figure_4](https://github.com/user-attachments/assets/fc18a452-f783-438c-93e8-f426f3b13159)
![Figure_5](https://github.com/user-attachments/assets/d1113e81-f36c-4d82-9546-6f8577ae301d)

# Feature Selection

![image](https://github.com/user-attachments/assets/ae8e3a5d-6d6e-49e1-bf64-ad31a279e448)

# Accuracy

![image](https://github.com/user-attachments/assets/66bc9540-829b-40b3-a2de-eb8fd7234ec3)

# Hyperparameter Tuning
Hyperparameter tuning is a crucial step in optimizing the performance of a machine 
learning model. It involves adjusting the parameters that govern the training process to 
find the optimal set of values that yield the best model performance.
Importance of Hyperparameter Tuning:
Enhances Model Accuracy: Proper tuning can significantly improve the
 model’s accuracy and predictive power.
Prevents Overfitting: By finding the right balance, hyperparameter tuning 
helps in preventing the model from overfitting to the training data.
Optimizes Performance: It ensures the model is both efficient and 
effective in its predictions.
Method Used: Custom Tuning with Random Forest
We implemented a custom hyperparameter tuning method for the Random Forest
classifier. The code iteratively tests different values for the max_depth parameter, which 
controls the maximum depth of each tree in the forest.
Process:
Parameter Initialization: We initialized the Random Forest classifier 
with a fixed random state to ensure reproducibility.
Train-Test Split: The data was split into training and testing sets with a  test size of 
23%.
Iterative Training and Evaluation: For each value of max_depth from 1 to 30,the model
 was trained and evaluated using precision, accuracy, training score, and  testing score.
Result Collection: The performance metrics for each max_depth value were collected 
and  analyzed.

# Confusion Matrix
![image](https://github.com/user-attachments/assets/99660ec9-1caf-4b7f-81c4-53d38ab21fc8)

# UI
![image](https://github.com/user-attachments/assets/ec9f755b-d2ba-455b-b6b5-4246e0a0d137)

# Result
![image](https://github.com/user-attachments/assets/33fb0521-cfa3-4d5c-95e7-119c98e743f8)

# Conclusion
This study aimed to develop a robust model to predict mobile price ranges based 
on various features. Through comprehensive data exploration, feature 
engineering, and rigorous model training and evaluation, we achieved significant
 insights and results.
Key Findings:
Data Exploration: We identified key features such as RAM, battery power, and
 pixel resolution that are highly correlated with the price range.
Visualization: Boxplots and pie charts provided a clear understanding of
 feature distributions and their impact on the target variable.
Model Training: A Random Forest classifier was used, achieving high accuracy
 and robust performance metrics.
Evaluation: The model was evaluated using accuracy, precision, recall, and F1
 score, confirming its reliability.

# Future Work:
Feature Enhancement: Including more advanced features such as user
 reviews and market trends could improve the model.
Model Optimization: Experimenting with other machine learning algorithms
 and deep learning techniques could enhance prediction accuracy.
Deployment: Developing a user-friendly application for real-time price
 prediction could provide practical benefits for consumers and manufacturers.
















