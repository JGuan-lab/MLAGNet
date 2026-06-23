# Read a txt file, sort by Average PSNR, and save the top five entries.
import config

def process_psnr_file(input_file, output_file):
    # Read file content and split into lines.
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Strip the header line.
    header = lines[0]
    data_lines = lines[1:]

    # Sort by Average PSNR descending and keep the top five.
    sorted_data = sorted(data_lines, key=lambda x: float(x.split()[1]), reverse=True)[:5]

    # Write to a new txt file.
    with open(output_file, 'w') as f:
        f.write(header)  # Write the header line.
        f.writelines(sorted_data)  # Write the top five sorted lines.

# Usage example.
input_file = config.SR_RESULT_TXT_2FULL_EVA  # Input file.
output_file = config.SR_RESULT_TXT_2FULL_FINAL  # Output file.
process_psnr_file(input_file, output_file)
