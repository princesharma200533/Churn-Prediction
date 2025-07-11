# Churn Prediction for ConnectSphere Telecom
# Complete Jupyter Notebook Implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("🚀 ConnectSphere Telecom Churn Prediction Project")
print("=" * 60)

# ============================================================================
# 1. DATA CREATION AND LOADING
# ============================================================================

print("\n📦 Step 1: Creating Sample Dataset")
print("-" * 40)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create sample dataset
n_customers = 2000

# Generate customer data
data = {
    'CustomerID': [f'CUST_{i:04d}' for i in range(1, n_customers + 1)],
    'CallDuration': np.random.normal(150, 50, n_customers).clip(10, 500),
    'DataUsage': np.random.exponential(5, n_customers).clip(0.1, 50),
    'ContractLength': np.random.choice([12, 24, 36], n_customers, p=[0.4, 0.4, 0.2]),
    'MonthlyCharges': np.random.normal(65, 20, n_customers).clip(20, 150),
    'PaymentMethod': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check', 'Mailed Check'], 
                                    n_customers, p=[0.3, 0.25, 0.25, 0.2])
}

# Calculate TotalCharges based on MonthlyCharges and ContractLength
data['TotalCharges'] = data['MonthlyCharges'] * (data['ContractLength'] / 12) * np.random.uniform(0.8, 1.2, n_customers)

# Create churn labels with realistic patterns
churn_probability = (
    0.1 +  # Base churn rate
    0.3 * (data['MonthlyCharges'] > 80) +  # High charges increase churn
    0.2 * (data['DataUsage'] < 2) +  # Low usage increases churn
    0.15 * (data['ContractLength'] == 12) +  # Short contracts increase churn
    0.1 * (np.array(data['PaymentMethod']) == 'Electronic Check')  # Payment method effect
)

data['Churn'] = np.random.binomial(1, churn_probability.clip(0, 1), n_customers)
data['Churn'] = ['Yes' if x == 1 else 'No' for x in data['Churn']]

# Create DataFrame
df = pd.DataFrame(data)

print(f"✅ Dataset created with {len(df)} customers")
print(f"📊 Dataset shape: {df.shape}")
print(f"🎯 Churn distribution:")
print(df['Churn'].value_counts())
print(f"📈 Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")

# Display first few rows
print("\n📋 First 5 rows of the dataset:")
print(df.head())

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n\n🔍 Step 2: Exploratory Data Analysis")
print("-" * 40)

# Basic statistics
print("📊 Dataset Info:")
print(df.info())

print("\n📈 Numerical Features Statistics:")
print(df.describe())

# Check for missing values
print(f"\n🔍 Missing values: {df.isnull().sum().sum()}")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

print("\n\n🛠️ Step 3: Data Preprocessing")
print("-" * 40)

# Create a copy for preprocessing
df_processed = df.copy()

# Remove CustomerID as it's not useful for prediction
df_processed = df_processed.drop('CustomerID', axis=1)

# Encode categorical variables
print("🔄 Encoding categorical variables...")

# Label encode the target variable
label_encoder = LabelEncoder()
df_processed['Churn_encoded'] = label_encoder.fit_transform(df_processed['Churn'])

# One-hot encode PaymentMethod
payment_dummies = pd.get_dummies(df_processed['PaymentMethod'], prefix='PaymentMethod')
df_processed = pd.concat([df_processed, payment_dummies], axis=1)
df_processed = df_processed.drop(['PaymentMethod', 'Churn'], axis=1)

print("✅ Categorical encoding completed")
print(f"📊 Processed dataset shape: {df_processed.shape}")

# Separate features and target
X = df_processed.drop('Churn_encoded', axis=1)
y = df_processed['Churn_encoded']

print(f"🎯 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Normalize numerical features
print("\n🔧 Normalizing features using MinMaxScaler...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling completed")

# ============================================================================
# 4. BUILD ARTIFICIAL NEURAL NETWORK
# ============================================================================

print("\n\n🧠 Step 4: Building Artificial Neural Network")
print("-" * 40)

# Build the ANN model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],), name='hidden_layer_1'),
    layers.Dense(32, activation='relu', name='hidden_layer_2'),
    layers.Dense(1, activation='sigmoid', name='output_layer')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("🏗️ Model Architecture:")
model.summary()

# ============================================================================
# 5. TRAIN THE MODEL
# ============================================================================

print("\n\n🚀 Step 5: Training the Model")
print("-" * 40)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("✅ Model training completed!")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================

print("\n\n📊 Step 6: Model Evaluation")
print("-" * 40)

# Make predictions
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("🎯 Model Performance Metrics:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives:  {cm[0,0]}")
print(f"   False Positives: {cm[0,1]}")
print(f"   False Negatives: {cm[1,0]}")
print(f"   True Positives:  {cm[1,1]}")

# Detailed classification report
print(f"\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

print("\n\n📈 Step 7: Creating Visualizations")
print("-" * 40)

# Plot training history
plt.figure(figsize=(15, 5))

# Training & Validation Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Training & Validation Loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Confusion Matrix Heatmap
plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'], 
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.tight_layout()
plt.show()

print("✅ Visualizations created successfully!")

# ============================================================================
# 8. BUSINESS INSIGHTS AND CSV EXPORT
# ============================================================================

print("\n\n💼 Step 8: Business Insights and Export")
print("-" * 40)

# Predict churn probabilities for entire dataset
X_all_scaled = scaler.transform(X)
churn_probabilities = model.predict(X_all_scaled).flatten()

# Create results DataFrame
results_df = df.copy()
results_df['Churn_Probability'] = churn_probabilities
results_df['Predicted_Churn'] = (churn_probabilities > 0.5).astype(int)
results_df['Predicted_Churn'] = results_df['Predicted_Churn'].map({0: 'No', 1: 'Yes'})

# Identify at-risk customers (probability > 0.5)
at_risk_customers = results_df[results_df['Churn_Probability'] > 0.5].copy()
at_risk_customers = at_risk_customers.sort_values('Churn_Probability', ascending=False)

print(f"🚨 At-risk customers identified: {len(at_risk_customers)}")
print(f"📊 Percentage of customers at risk: {len(at_risk_customers)/len(results_df):.2%}")

# Display top 10 at-risk customers
print(f"\n🔝 Top 10 customers with highest churn probability:")
print(at_risk_customers[['CustomerID', 'MonthlyCharges', 'DataUsage', 'ContractLength', 'Churn_Probability']].head(10))

# Export at-risk customers to CSV
at_risk_customers.to_csv('at_risk_customers.csv', index=False)
print(f"\n💾 At-risk customers exported to 'at_risk_customers.csv'")

# Business insights
print(f"\n💡 Key Business Insights:")
print(f"   • {len(at_risk_customers)} customers are at high risk of churning")
print(f"   • Average churn probability among at-risk customers: {at_risk_customers['Churn_Probability'].mean():.3f}")
print(f"   • Average monthly charges of at-risk customers: ${at_risk_customers['MonthlyCharges'].mean():.2f}")
print(f"   • Most common contract length among at-risk customers: {at_risk_customers['ContractLength'].mode().iloc[0]} months")

# Model performance summary
print(f"\n🎯 Model Performance Summary:")
print(f"   • The ANN model achieved {accuracy:.2%} accuracy on the test set")
print(f"   • Precision: {precision:.2%} (of predicted churners, {precision:.2%} actually churned)")
print(f"   • Recall: {recall:.2%} (of actual churners, {recall:.2%} were correctly identified)")
print(f"   • F1-Score: {f1:.3f} (balanced measure of precision and recall)")

print(f"\n🎉 Churn Prediction Project Completed Successfully!")
print("=" * 60)
