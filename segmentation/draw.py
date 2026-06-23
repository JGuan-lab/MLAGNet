# import matplotlib.pyplot as plt
# import re

# # Define a function to read txt file and parse Epoch and Accuracy
# def parse_log_file(file_path):
#     epochs = []
#     accuracies = []

#     # Use regex to extract EPOCH and ACCURACY
#     pattern = r"Epoch【(\d+)/\d+】.*Accuracy:(\d+\.\d+)"

#     with open(file_path, 'r') as file:
#         for line in file:
#             match = re.search(pattern, line)
#             if match:
#                 # Extract Epoch and Accuracy
#                 epoch = int(match.group(1))
#                 accuracy = float(match.group(2))
#                 epochs.append(epoch)
#                 accuracies.append(accuracy)
#
#     return epochs, accuracies

# # Plot line chart
# def plot_accuracy(epochs, accuracies):
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, accuracies, marker='o', color='b', label='Accuracy')

#     # Set chart title and labels
#     plt.title('Accuracy over Epochs', fontsize=16)
#     plt.xlabel('Epoch', fontsize=14)
#     plt.ylabel('Accuracy', fontsize=14)

#     # Add grid and legend
#     plt.grid(True)
#     plt.legend()

#     # Display chart
#     plt.show()

# # Main function
# if __name__ == "__main__":
#     # Replace with your txt file path
#     file_path = '/home/Yb/uureal/fenge/result/train_log.txt'  # Replace this path with your file path

#     # Parse file and plot
#     epochs, accuracies = parse_log_file(file_path)
#     plot_accuracy(epochs, accuracies)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import matplotlib.pyplot as plt
import re

# Define a function to read txt file and parse Epoch and Accuracy
def parse_log_file(file_path):
    epochs = []
    accuracies = []

    # Use regex to extract EPOCH and ACCURACY
    pattern = r"Epoch【(\d+)/\d+】.*Accuracy:(\d+\.\d+)"

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Extract Epoch and Accuracy
                epoch = int(match.group(1))
                accuracy = float(match.group(2))
                epochs.append(epoch)
                accuracies.append(accuracy)

    return epochs, accuracies

# Plot line chart and save to specified path
def plot_accuracy(epochs, accuracies, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o', color='b', label='Accuracy')

    # Set chart title and labels
    plt.title('Accuracy over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Save image to specified path
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

# Main function
if __name__ == "__main__":
    # Replace with your txt file path
    file_path = config.DRAW_LOG_FILE  # Replace this path with your log file path

    # Replace with your desired image save path
    save_path = config.DRAW_PLOT_OUTPUT  # Replace with your desired image save path

    # Parse file and save image
    epochs, accuracies = parse_log_file(file_path)
    plot_accuracy(epochs, accuracies, save_path)
