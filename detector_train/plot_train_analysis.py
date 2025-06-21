import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def parse_log_file(log_text):
    """Parse the training log file to extract epoch, training loss, validation loss, and time metrics."""
    
    # Regular expressions to extract information
    epoch_pattern = r"Epoch: \[(\d+)\]\[(\d+)/(\d+)\] Time ([\d\.]+) \(([\d\.]+)\) Data ([\d\.]+) \(([\d\.]+)\) Loss ([\d\.]+)"
    validation_pattern = r"Validation: \[(\d+)\]\[(\d+)/(\d+)\] Time ([\d\.]+) \(([\d\.]+)\) Loss ([\d\.]+)"
    
    # Initialize dictionaries to store data
    data = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'train_times': [],
        'data_loading_times': [],
        'val_times': []
    }
    
    # Split the log text into lines
    lines = log_text.strip().split('\n')
    
    for line in lines:
        # Extract training information
        train_match = re.search(epoch_pattern, line)
        if train_match:
            epoch = int(train_match.group(1))
            batch = int(train_match.group(2))
            total_batches = int(train_match.group(3))
            batch_time = float(train_match.group(4))
            avg_time = float(train_match.group(5))
            data_time = float(train_match.group(6))
            avg_data_time = float(train_match.group(7))
            loss = float(train_match.group(8))
            
            data['epochs'].append(epoch)
            data['train_losses'].append(loss)
            data['train_times'].append(batch_time)
            data['data_loading_times'].append(data_time)
            
        # Extract validation information
        val_match = re.search(validation_pattern, line)
        if val_match:
            # We already have the epoch from the training line
            val_batch = int(val_match.group(2))
            val_total_batches = int(val_match.group(3))
            val_time = float(val_match.group(4))
            val_avg_time = float(val_match.group(5))
            val_loss = float(val_match.group(6))
            
            data['val_losses'].append(val_loss)
            data['val_times'].append(val_time)
            
    return data

def visualize_loss_curves(data):
    """Generate visualizations for the loss curves."""
    
    epochs = data['epochs']
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    
    plt.figure(figsize=(12, 10))
    
    # 1. Training and Validation Loss Curve
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', marker='o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', marker='x', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 2. Log-scale Loss Curve
    plt.subplot(2, 2, 2)
    plt.semilogy(epochs, train_losses, 'b-', marker='o', label='Training Loss')
    plt.semilogy(epochs, val_losses, 'r-', marker='x', label='Validation Loss')
    plt.title('Log-scale Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 3. Loss Improvement Rate
    train_improvement = [0] + [(train_losses[i-1] - train_losses[i]) for i in range(1, len(train_losses))]
    val_improvement = [0] + [(val_losses[i-1] - val_losses[i]) for i in range(1, len(val_losses))]
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_improvement, 'b-', marker='o', label='Training Improvement')
    plt.plot(epochs, val_improvement, 'r-', marker='x', label='Validation Improvement')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Loss Improvement per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Reduction')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 4. Training-Validation Loss Difference
    loss_diff = [t - v for t, v in zip(train_losses, val_losses)]
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, loss_diff, 'g-', marker='o')
    plt.title('Training-Validation Loss Difference')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss - Val Loss')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig('training_loss_visualizations.png', dpi=300)
    plt.show()

def visualize_time_metrics(data):
    """Generate visualizations for the time metrics."""
    
    epochs = data['epochs']
    train_times = data['train_times']
    data_loading_times = data['data_loading_times']
    val_times = data['val_times']
    
    plt.figure(figsize=(12, 8))
    
    # 1. Training and Validation Times
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_times, 'b-', marker='o', label='Training Time (per batch)')
    plt.plot(epochs, val_times, 'r-', marker='x', label='Validation Time (per batch)')
    plt.title('Training and Validation Times per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 2. Data Loading Time vs. Processing Time
    processing_times = [t - d for t, d in zip(train_times, data_loading_times)]
    
    plt.subplot(2, 1, 2)
    plt.bar(epochs, data_loading_times, width=0.4, align='edge', label='Data Loading Time')
    plt.bar([e + 0.4 for e in epochs], processing_times, width=0.4, align='edge', label='Processing Time', color="red")
    plt.title('Data Loading vs Processing Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, axis='y')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig('training_time_visualizations.png', dpi=300)
    plt.show()

def visualize_early_stopping(data):
    """Generate visualization for early stopping analysis."""
    
    epochs = data['epochs']
    val_losses = data['val_losses']
    
    plt.figure(figsize=(10, 6))
    
    # Smoothed validation loss for early stopping analysis
    window_size = min(3, len(val_losses))
    smoothed_val_losses = []
    
    for i in range(len(val_losses)):
        if i < window_size - 1:
            # For the first few epochs, use available data
            smoothed_val_losses.append(np.mean(val_losses[:i+1]))
        else:
            # For later epochs, use the window
            smoothed_val_losses.append(np.mean(val_losses[i-window_size+1:i+1]))
    
    plt.plot(epochs, val_losses, 'r-', marker='x', alpha=0.7, label='Validation Loss')
    plt.plot(epochs, smoothed_val_losses, 'r--', label=f'Smoothed Val Loss (window={window_size})')
    
    # Find the epoch with minimum validation loss
    min_loss_epoch = epochs[np.argmin(val_losses)]
    min_loss_value = min(val_losses)
    
    plt.axvline(x=min_loss_epoch, color='g', linestyle='--', label=f'Min Val Loss at Epoch {min_loss_epoch}')
    plt.axhline(y=min_loss_value, color='g', linestyle=':', alpha=0.5)
    
    plt.title('Early Stopping Analysis')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('early_stopping_analysis.png', dpi=300)
    plt.show()

def analyze_training_progression(data):
    """Analyze the training progression and print insights."""
    
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    epochs = data['epochs']
    
    # Training progression insights
    print("===== Training Progress Analysis =====")
    print(f"Initial training loss: {train_losses[0]:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Loss reduction: {train_losses[0] - train_losses[-1]:.4f} ({(1 - train_losses[-1]/train_losses[0])*100:.2f}%)")
    
    print(f"\nInitial validation loss: {val_losses[0]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Loss reduction: {val_losses[0] - val_losses[-1]:.4f} ({(1 - val_losses[-1]/val_losses[0])*100:.2f}%)")
    
    best_val_epoch = epochs[np.argmin(val_losses)]
    best_val_loss = min(val_losses)
    print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_val_epoch}")
    
    if train_losses[-1] > val_losses[-1]:
        print("\nModel might be underfitting (training loss > validation loss)")
    elif train_losses[-1] < val_losses[-1]:
        diff = val_losses[-1] - train_losses[-1]
        if diff > 0.1:
            print(f"\nModel might be overfitting (validation loss > training loss by {diff:.4f})")
        else:
            print("\nModel seems well-balanced (training and validation losses are close)")
    
    val_window = min(3, len(val_losses))
    recent_val_trend = [val_losses[-i-1] - val_losses[-i] for i in range(val_window-1)]
    
    if all(trend < 0 for trend in recent_val_trend):
        print("\nValidation loss is consistently increasing in recent epochs - consider early stopping.")
    elif all(trend > 0 for trend in recent_val_trend):
        print("\nValidation loss is still consistently decreasing - training could continue.")
    else:
        print("\nValidation loss is fluctuating in recent epochs.")

    total_epochs = max(epochs) + 1
    target_epochs = 399
    training_progress = (total_epochs / target_epochs) * 100
    print(f"\nTraining progress: {training_progress:.2f}% ({total_epochs}/{target_epochs} epochs)")

    avg_epoch_time = np.mean(data['train_times']) + np.mean(data['val_times'])
    remaining_time = (target_epochs - total_epochs) * avg_epoch_time
    print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
    print(f"Estimated remaining time: {remaining_time/60:.2f} minutes ({remaining_time/3600:.2f} hours)")

def main():
    with open('training_log.txt', 'r') as f:
        log_text = f.read()

    data = parse_log_file(log_text)

    visualize_loss_curves(data)
    visualize_time_metrics(data)
    visualize_early_stopping(data)

    analyze_training_progression(data)

if __name__ == "__main__":
    main()