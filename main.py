# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import fetch_openml

print("Loading MNIST dataset...")
# Load the MNIST dataset with cache disabled to avoid checksum issues
mnist = fetch_openml('mnist_784', version=1, cache=False)

# Convert to numpy arrays for efficiency
X = mnist.data.to_numpy() if hasattr(mnist.data, 'to_numpy') else np.array(mnist.data)
y = mnist.target.astype(int)  # Convert target to integer

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Sample a subset for faster training (optional: remove these lines to use full dataset)
print("Sampling subset for faster training...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=5000, random_state=42, stratify=y)
X, y = X_sample, y_sample
print(f"Using {X.shape[0]} samples for training and testing")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Standardize the feature variables
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models with optimized hyperparameters
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
    "SVC": SVC(kernel='rbf', C=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
}

# Train and evaluate each model
results = {}
training_times = {}

print("\n" + "=" * 70)
print("TRAINING AND EVALUATING MODELS")
print("=" * 70)

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Time the training process
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    training_times[name] = training_time

    print(f"  Training time: {training_time:.2f} seconds")

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred)

    # Store the results
    results[name] = {
        "Accuracy": accuracy,
        "Classification Report": report,
        "Confusion Matrix": confusion,
        "Training Time": training_time
    }

    print(f"  Accuracy: {accuracy:.4f}")

# Create a DataFrame for the summary accuracy results
summary_df = pd.DataFrame({
    name: {
        "Accuracy": metrics["Accuracy"],
        "F1-Score": metrics["Classification Report"]["weighted avg"]["f1-score"],
        "Precision": metrics["Classification Report"]["weighted avg"]["precision"],
        "Recall": metrics["Classification Report"]["weighted avg"]["recall"],
        "Training Time (s)": metrics["Training Time"]
    }
    for name, metrics in results.items()
}).T

# Display the summary DataFrame
print("\n" + "=" * 70)
print("MNIST DATASET MODEL PERFORMANCE SUMMARY")
print("=" * 70)
print(summary_df.to_string())

# Save summary to CSV
summary_df.to_csv('mnist_model_comparison.csv')
print("\nSummary saved to 'mnist_model_comparison.csv'")

# Output detailed classification reports
print("\n" + "=" * 70)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 70)

for name, metrics in results.items():
    print(f"\n{'=' * 70}")
    print(f"Model: {name}")
    print(f"{'=' * 70}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Training Time: {metrics['Training Time']:.2f} seconds")

    print("\nPer-Class Metrics:")
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)

    for label in sorted([k for k in metrics["Classification Report"].keys()
                         if k not in ["accuracy", "macro avg", "weighted avg"]]):
        score = metrics["Classification Report"][label]
        print(f"{label:<8} {score['precision']:<12.4f} {score['recall']:<12.4f} "
              f"{score['f1-score']:<12.4f} {int(score['support']):<10}")

    print("\nConfusion Matrix:")
    print(metrics["Confusion Matrix"])

# Visualization 1: Accuracy and F1-scores comparison
model_names = list(results.keys())
accuracies = [metrics["Accuracy"] for metrics in results.values()]
f1_scores = [metrics["Classification Report"]["weighted avg"]["f1-score"] for metrics in results.values()]

# Set up bar width and positions
bar_width = 0.35
index = np.arange(len(model_names))

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# Subplot 1: Accuracy vs F1-Score
ax1 = plt.subplot(2, 2, 1)
bar1 = ax1.bar(index, accuracies, bar_width, label='Accuracy', color='lightgreen', alpha=0.8)
bar2 = ax1.bar(index + bar_width, f1_scores, bar_width, label='F1-Score', color='orange', alpha=0.8)

# Add value labels on bars
for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

ax1.set_xlabel('Machine Learning Models', fontsize=11)
ax1.set_ylabel('Scores', fontsize=11)
ax1.set_title('Model Performance: Accuracy vs F1-Score', fontsize=13, fontweight='bold')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(model_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1.1])

# Subplot 2: Training Time Comparison
ax2 = plt.subplot(2, 2, 2)
times = [metrics["Training Time"] for metrics in results.values()]
bars = ax2.bar(model_names, times, color='skyblue', alpha=0.8)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

ax2.set_xlabel('Machine Learning Models', fontsize=11)
ax2.set_ylabel('Training Time (seconds)', fontsize=11)
ax2.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
ax2.set_xticklabels(model_names, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# Subplot 3: Precision and Recall Comparison
ax3 = plt.subplot(2, 2, 3)
precisions = [metrics["Classification Report"]["weighted avg"]["precision"] for metrics in results.values()]
recalls = [metrics["Classification Report"]["weighted avg"]["recall"] for metrics in results.values()]

bar3 = ax3.bar(index, precisions, bar_width, label='Precision', color='lightcoral', alpha=0.8)
bar4 = ax3.bar(index + bar_width, recalls, bar_width, label='Recall', color='lightblue', alpha=0.8)

# Add value labels
for bars in [bar3, bar4]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

ax3.set_xlabel('Machine Learning Models', fontsize=11)
ax3.set_ylabel('Scores', fontsize=11)
ax3.set_title('Model Performance: Precision vs Recall', fontsize=13, fontweight='bold')
ax3.set_xticks(index + bar_width / 2)
ax3.set_xticklabels(model_names, rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 1.1])

# Subplot 4: Overall Performance Radar Chart
ax4 = plt.subplot(2, 2, 4, projection='polar')

# Find best model for highlighting
best_model_idx = accuracies.index(max(accuracies))

categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for idx, name in enumerate(model_names):
    values = [
        results[name]["Accuracy"],
        results[name]["Classification Report"]["weighted avg"]["precision"],
        results[name]["Classification Report"]["weighted avg"]["recall"],
        results[name]["Classification Report"]["weighted avg"]["f1-score"]
    ]
    values += values[:1]

    linewidth = 3 if idx == best_model_idx else 1.5
    alpha = 0.8 if idx == best_model_idx else 0.4

    ax4.plot(angles, values, 'o-', linewidth=linewidth, label=name, alpha=alpha)
    ax4.fill(angles, values, alpha=0.15)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=10)
ax4.set_ylim(0, 1)
ax4.set_title('Overall Performance Comparison', fontsize=13, fontweight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax4.grid(True)

plt.tight_layout()
plt.savefig('mnist_performance_comparison.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to 'mnist_performance_comparison.png'")
plt.show()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"Best performing model: {model_names[best_model_idx]} "
      f"(Accuracy: {accuracies[best_model_idx]:.4f})")
print(f"Fastest training model: {model_names[times.index(min(times))]} "
      f"(Time: {min(times):.2f}s)")
