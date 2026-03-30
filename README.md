Linear Regression with Gradient Descent

This repository contains a simple Python program that demonstrates how to train a linear regression model using gradient descent.

Overview

The program learns the relationship between study hours and exam scores. It starts with a line y = m*x + b where both m (slope) and b (intercept) are initialized to 0. Through repeated updates, the model adjusts these parameters to minimize the error between predicted and actual scores.

How It Works

Training Data: Study hours (x_data) and exam scores (y_data).

Model: A straight line y = m*x + b.

Error Calculation: Difference between predicted and actual scores.

Gradient Accumulation: Collects how each data point suggests adjusting slope and intercept.

Parameter Update: Adjusts m and b using the averaged gradients and a learning rate.

Progress Tracking: Prints loss, slope, and intercept every 100 epochs.

Example Output

Epoch 0: Loss = 2262.2400, m = 4.0000, b = 1.2000
Epoch 100: Loss = 58.7400, m = 14.9756, b = 12.0364
Epoch 500: Loss = 3.9100, m = 11.2839, b = 25.3648
Epoch 1000: Loss = 0.1300, m = 10.2361, b = 29.1476

Final Model

After training, the model converges to approximately:

y ≈ 10x + 29

This means each additional hour of study adds about 10 points to the exam score.

Usage

Run the program with:

python linear_regression.py

It will train the model and print progress along the way, then predict the score for 6 hours of study.

Intuition

Think of the program as a student learning from a teacher:

The student guesses scores based on hours.

The teacher shows the real scores.

The student adjusts their rule each round.

After enough practice, the student predicts accurately.

License

This project is open source under the MIT License.
