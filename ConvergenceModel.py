import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Load and combine datasets
df1 = pd.read_csv('pdmm_optimization_results_2.csv')
df2 = pd.read_csv('pdmm_optimization_results_3.csv')
df = pd.concat([df1, df2], ignore_index=True)

# Drop irrelevant columns  and prepare data
df = df.drop(columns=['Final_Error'])  # Keep only features and target
X = df[['Timer','K_decision', 'Rejection_Threshold']].values
y = df['Converged'].astype(int).values  # Convert boolean to 0/1

# Handle class imbalance (upsample minority class)
df_majority = df[df['Converged'] == False]
df_minority = df[df['Converged'] == True]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced[['Timer','K_decision', 'Rejection_Threshold']].values
y = df_balanced['Converged'].astype(int).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()])

# Train model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=1)

# Evaluate
test_loss, test_acc, test_auc = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")

# Predict convergence for new parameters
def predict_convergence(timer,k_decision, rejection_threshold):
    input_data = np.array([[timer,k_decision, rejection_threshold]])
    scaled_input = scaler.transform(input_data)
    prob = model.predict(scaled_input)[0][0]
    return prob >= 0.5  # Threshold at 0.5

# Example: Predict for K=1, Threshold=5.5
print(predict_convergence(1, 5.5))  # Output: True/False

# Generate a grid of parameters to explore
k_values = np.arange(1, 5)  # K_decision ranges from 1-4
thresholds = np.linspace(1, 15, 50)  # Thresholds from 1 to 15

converged_params = []
for k in k_values:
    for t in thresholds:
        if predict_convergence(k, t):
            converged_params.append((k, t))

print("Suggested parameters for convergence:")
print(np.unique(converged_params, axis=0))