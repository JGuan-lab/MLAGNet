# import re

# # Define a function to read the file and extract IoU values.
# def extract_max_iou(filename):
#     max_iou =0  # Initialize the maximum IoU to zero.
#     with open(filename, 'r') as file:
#         for line in file:
#             # Use a regular expression to extract the IoU value.
#             match = re.search(r'Average PSNR: (\d+\.\d+)', line)
#             if match:
#                 iou_value = float(match.group(1))  # Convert the matched IoU value to a float.
#                 if iou_value > max_iou:
#                     max_iou = iou_value  # Update the maximum IoU value.
#     return max_iou

# # Input filename.
# filename = '/home/Yb/uureal/result/txt/8fullgan_eva_results.txt'  # Replace with your txt file path.

# # Call the function and print the result.
# max_iou = extract_max_iou(filename)
# print(f'The highest IoU is: {max_iou}')
import re

def extract_max_average_psnr(filename):
    max_psnr = 0  # Initialize the maximum Average PSNR to zero.
    max_psnr_line = ""  # Used to store the corresponding line.
    with open(filename, 'r') as file:
        for line in file:
            # Skip the header row.
            if line.startswith("Model Name"):
                continue
            # Split each line into columns.
            columns = line.strip().split("\t")
            if len(columns) >= 3:  # Ensure the line has at least 3 columns.
                try:
                    average_psnr = float(columns[1])  # The second column is Average PSNR.
                    if average_psnr > max_psnr:
                        max_psnr = average_psnr  # Update the maximum value.
                        max_psnr_line = line.strip()  # Save this line.
                except ValueError:
                    continue  # Ignore content that cannot be converted to a float.
    return max_psnr, max_psnr_line

# Input filename.
import config
filename = config.SR_RESULT_TXT_RANK  # Replace with your txt file path.

# Call the function and print the result.
max_psnr, max_psnr_line = extract_max_average_psnr(filename)
print(f"The highest Average PSNR is: {max_psnr}")
print(f"The corresponding line is: {max_psnr_line}")
