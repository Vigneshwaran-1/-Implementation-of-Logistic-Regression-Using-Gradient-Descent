# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary .
4. Calculate the y-prediction.

## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: P.VIGNESHWARAN
Register Number: 212224040358


import numpy as np
import matplotlib.pyplot as plt

# Dataset: CGPA, Aptitude Score
X = np.array([
    [6.0, 45], [6.5, 50], [7.0, 55], [7.2, 58], [7.5, 60],
    [8.0, 65], [8.2, 68], [8.5, 72], [9.0, 80], [9.2, 85]
])

# Placement: 0 = Not Placed, 1 = Placed
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Normalize the input features
mean = X.mean(axis=0)
std = X.std(axis=0)
X_scaled = (X - mean) / std

# Add bias column
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize weights
weights = np.zeros(X_scaled.shape[1])

# Gradient Descent
learning_rate = 0.1
epochs = 1000

for i in range(epochs):
    z = np.dot(X_scaled, weights)
    prediction = sigmoid(z)

    gradient = np.dot(X_scaled.T, (prediction - y)) / len(y)
    weights = weights - learning_rate * gradient

# Predict training data
probability = sigmoid(np.dot(X_scaled, weights))
predicted = (probability >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(predicted == y) * 100

print("Actual Values:   ", y)
print("Predicted Values:", predicted)
print("Accuracy:", round(accuracy, 2), "%")

# New student prediction
cgpa = float(input("\nEnter CGPA: "))
aptitude = float(input("Enter Aptitude Score: "))

new_data = np.array([(cgpa - mean[0]) / std[0],
                     (aptitude - mean[1]) / std[1]])

new_data = np.insert(new_data, 0, 1)

prob = sigmoid(np.dot(new_data, weights))

print("Placement Probability:", round(prob * 100, 2), "%")

if prob >= 0.5:
    print("Placement Status: Placed")
else:
    print("Placement Status: Not Placed")

# Visualization
plt.figure(figsize=(8, 5))

plt.scatter(X[y == 0, 0], X[y == 0, 1],
            label="Not Placed", marker="o")

plt.scatter(X[y == 1, 0], X[y == 1, 1],
            label="Placed", marker="x")

plt.scatter(cgpa, aptitude, s=100,
            label="New Student", marker="*")

plt.xlabel("CGPA")
plt.ylabel("Aptitude Score")
plt.title("Logistic Regression using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

    
```
## OUTPUT
<img width="743" height="410" alt="image" src="https://github.com/user-attachments/assets/75f15352-036c-459c-8029-19ba4c6fbe8e" />




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

