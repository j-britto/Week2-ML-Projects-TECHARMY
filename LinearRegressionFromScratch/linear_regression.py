import numpy as np

# Sample dataset (House area in square feet and price in lakh)
x_sqft = np.array([1000, 1200, 1500, 1800])
y_lakh = np.array([30, 35, 45, 55])

# Calculate means
x_mean = np.mean(x_sqft)
y_mean = np.mean(y_lakh)

# Calculate slope (m) and intercept (b)
numerator = np.sum((x_sqft - x_mean) * (y_lakh - y_mean))
denominator = np.sum((x_sqft - x_mean) ** 2)
m = numerator / denominator
b = y_mean - (m * x_mean)

print(f"Linear Regression Equation:")
print(f"y = {m:.4f}x + ({b:.4f})")

# Predict house price for 2000 sqft
x_new = 2000
y_pred = m * x_new + b
print(f"\nPredicted price for {x_new} sqft = {y_pred:.2f} lakh")

# Predict all training values
predictions = m * x_sqft + b
print("\nPredictions on training data:")
for area, pred in zip(x_sqft, predictions):
    print(f"{area} sqft --> {pred:.2f} lakh")

# Mean Squared Error
mse = np.mean((y_lakh - predictions) ** 2)
print(f"\nMean Squared Error (MSE): {mse:.4f}")
