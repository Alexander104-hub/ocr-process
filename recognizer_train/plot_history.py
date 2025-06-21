import matplotlib.pyplot as plt
import seaborn as sns
import json

with open('training_history.json', 'r') as f:
    history = json.load(f)

# Set global style parameters for all plots
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Extract data from history
epochs = range(1, len(history['loss']) + 1)

# 1. Plot training and validation loss
plt.figure()
plt.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
plt.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Plot training and validation CER metric
plt.figure()
plt.plot(epochs, history['CER_metric'], 'g-', linewidth=2, label='Training CER')
plt.plot(epochs, history['val_CER_metric'], 'purple', linewidth=2, label='Validation CER')
plt.title('Training and Validation Character Error Rate (CER)')
plt.xlabel('Epochs')
plt.ylabel('CER')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Plot CER metric with standard deviation bands using seaborn
plt.figure()
sns.lineplot(x=epochs, y=history['CER_metric'], label='Training CER', linewidth=2)
sns.lineplot(x=epochs, y=history['val_CER_metric'], label='Validation CER', linewidth=2)
plt.title('Character Error Rate (CER) with Confidence Bands')
plt.xlabel('Epochs')
plt.ylabel('CER')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Plot learning rate changes
plt.figure()
plt.plot(epochs, history['learning_rate'], 'orange', linewidth=2)
plt.title('Learning Rate Schedule')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.yscale('log')  # Log scale for better visualization of learning rate changes
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Smoothed training curves
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.figure()
plt.plot(epochs, smooth_curve(history['loss']), 'b-', linewidth=2, label='Smoothed Training Loss')
plt.plot(epochs, smooth_curve(history['val_loss']), 'r-', linewidth=2, label='Smoothed Validation Loss')
plt.title('Smoothed Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Combined Loss and CER
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=color)
ax1.plot(epochs, history['loss'], color=color, linewidth=2, linestyle='-', label='Training Loss')
ax1.plot(epochs, history['val_loss'], color='tab:cyan', linewidth=2, linestyle='-', label='Validation Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('CER', color=color)
ax2.plot(epochs, history['CER_metric'], color=color, linewidth=2, linestyle='-', label='Training CER')
ax2.plot(epochs, history['val_CER_metric'], color='tab:orange', linewidth=2, linestyle='-', label='Validation CER')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title('Training and Validation Metrics')
plt.grid(True)
fig.tight_layout()
plt.show()

# 7. Heatmap of correlation between metrics
correlation_data = {
    'Training Loss': history['loss'],
    'Validation Loss': history['val_loss'],
    'Training CER': history['CER_metric'],
    'Validation CER': history['val_CER_metric'],
    'Learning Rate': history['learning_rate']
}

import pandas as pd
corr_df = pd.DataFrame(correlation_data)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Between Training Metrics')
plt.tight_layout()
plt.show()

# 8. Ratio of validation to training metrics
plt.figure()
val_train_loss_ratio = [v / t if t != 0 else 0 for v, t in zip(history['val_loss'], history['loss'])]
val_train_cer_ratio = [v / t if t != 0 else 0 for v, t in zip(history['val_CER_metric'], history['CER_metric'])]

plt.plot(epochs, val_train_loss_ratio, 'b-', linewidth=2, label='Val/Train Loss Ratio')
plt.plot(epochs, val_train_cer_ratio, 'r-', linewidth=2, label='Val/Train CER Ratio')
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
plt.title('Validation/Training Metrics Ratio')
plt.xlabel('Epochs')
plt.ylabel('Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Box plots for distribution analysis
plt.figure(figsize=(14, 7))
metrics_df = pd.DataFrame({
    'Training Loss': history['loss'],
    'Validation Loss': history['val_loss'],
    'Training CER': history['CER_metric'],
    'Validation CER': history['val_CER_metric']
})
sns.boxplot(data=metrics_df)
plt.title('Distribution of Training Metrics')
plt.ylabel('Value')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 10. Detailed training vs validation loss
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.plot(epochs, history['loss'], 'b-', linewidth=2)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(epochs, history['val_loss'], 'r-', linewidth=2)
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.show()

# 11. Detailed training vs validation CER
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.plot(epochs, history['CER_metric'], 'g-', linewidth=2)
plt.title('Training CER')
plt.xlabel('Epochs')
plt.ylabel('CER')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(epochs, history['val_CER_metric'], 'purple', linewidth=2)
plt.title('Validation CER')
plt.xlabel('Epochs')
plt.ylabel('CER')
plt.grid(True)

plt.tight_layout()
plt.show()

# 12. Histogram of improvement between epochs
loss_improvements = [history['loss'][i] - history['loss'][i+1] for i in range(len(history['loss'])-1)]
cer_improvements = [history['CER_metric'][i] - history['CER_metric'][i+1] for i in range(len(history['CER_metric'])-1)]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(loss_improvements, bins=20, alpha=0.7, color='blue')
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Distribution of Loss Improvements')
plt.xlabel('Loss Improvement (Epoch to Epoch)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(cer_improvements, bins=20, alpha=0.7, color='green')
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Distribution of CER Improvements')
plt.xlabel('CER Improvement (Epoch to Epoch)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()