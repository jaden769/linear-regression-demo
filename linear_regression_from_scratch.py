# Training data
x_data = [1, 2, 3, 4, 5]
y_data = [40, 50, 60, 70, 80]

# Model parameters
m = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Training loop
for epoch in range(epochs):
    gradient_m = 0
    gradient_b = 0
    n = len(x_data)

    # Go through every data point
    for i in range(n):
        x = x_data[i]
        y = y_data[i]

        # Prediction
        y_pred = m * x + b

        # Error
        error = y_pred - y

        # Accumulate gradients
        gradient_m += error * x
        gradient_b += error
    
    # Average gradients
    gradient_m = (2 / n) * gradient_m
    gradient_b = (2 / n) * gradient_b

    # Update parameters
    m = m - learning_rate * gradient_m
    b = b - learning_rate * gradient_b

    # Print progress sometimes
    if epoch % 100 == 0:
        total_loss = 0
        for i in range(n):
            y_pred = m * x_data[i] + b
            total_loss += (y_pred - y_data[i]) ** 2
        total_loss = total_loss / n
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}, m = {m:.4f}, b = {b:.4f}")

# Final model
print("\nTraining complete.")
print(f"Final m = {m:.4f}")
print(f"Final b = {b:.4f}")

# Test prediction
study_hours = 6
predicted_score = m * study_hours + b
print(f"Predicted score for {study_hours} hours: {predicted_score:.2f}")
