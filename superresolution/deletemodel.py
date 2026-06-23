import os
import config

# Read model names from a txt file
def get_model_names(txt_file):
    model_names = set()
    with open(txt_file, 'r') as f:
        lines = f.readlines()[1:]  # skip the header line
        for line in lines:
            model_name = line.split()[0]  # get the first value on each line, i.e. the model name
            model_names.add(model_name)
    return model_names

# Delete models not listed in the txt file
def delete_unused_models(model_folder, valid_model_names):
    for model_file in os.listdir(model_folder):
        if model_file not in valid_model_names:
            model_path = os.path.join(model_folder, model_file)
            if os.path.isfile(model_path):
                os.remove(model_path)
                print(f"Deleted: {model_file}")

# Example usage
txt_file = config.SR_DELETE_TXT  # your txt file
model_folder = config.SR_DELETE_MODEL_FOLDER  # path to the model folder

# get model names from the txt file
valid_model_names = get_model_names(txt_file)

# delete models not listed in the txt file
delete_unused_models(model_folder, valid_model_names)
