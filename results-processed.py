import os
import pandas as pd

# Define a function to extract information from the file content
def extract_info_from_file(file_path):
    with open(file_path) as file:
        lines = file.readlines()

    # Extract total time and final accuracy
    total_time = None
    final_accuracy = None
    for line in reversed(lines):
        if 'Total training time:' in line:
            total_time = float(line.strip().split(': ')[1].split(' ')[0])
        if 'accuracy:' in line:
            final_accuracy = float(line.strip().split(': ')[1].replace('%', ''))
            break  # Stop after the final accuracy is found as it is the first instance from the end

    return total_time, final_accuracy

# Function to parse file name
def parse_filename(file_name):
    parts = file_name.split('-')
    if len(parts) >= 4:
        optimizer = parts[0]
        model = parts[1]
        dataset = parts[2]
        augmentation = parts[3].split('.')[0]  # Remove file extension, prob need to add run number here
    else:
        optimizer, model, dataset, augmentation = None, None, None, None

    return optimizer, model, dataset, augmentation

# Specify the directory
folder_path = 'logs'  # Change 'folder_name' to the name of your folder

# DataFrame to store the results
results_df = pd.DataFrame(columns=['Optimizer', 'Model Architecture', 'Dataset', 'Data Augmentation', 'Total Time (s)', 'Final Accuracy (%)'])

# Process each file in the directory
for file in os.listdir(folder_path):
    if file.endswith('.out'):  # Check if the file is a .out file
        file_path = os.path.join(folder_path, file)
        total_time, final_accuracy = extract_info_from_file(file_path)
        optimizer, model, dataset, augmentation = parse_filename(file)
        
        # Append the results to the DataFrame
        results_df = results_df._append({
            'Optimizer': optimizer,
            'Model Architecture': model,
            'Dataset': dataset,
            'Data Augmentation': augmentation,
            'Final Accuracy (%)': final_accuracy,
            'Total Time (s)': total_time
        }, ignore_index=True)


results_df['Total Time (min)'] = results_df['Total Time (s)'] / 60

print(results_df)

grouped_df = results_df.groupby(['Optimizer', 'Model Architecture', 'Dataset', 'Data Augmentation']).agg({
    'Final Accuracy (%)': ['mean', 'std'],
    'Total Time (s)': ['mean', 'std'],
    'Total Time (min)': ['mean', 'std']
}).reset_index()

# Rename the columns for clarity
grouped_df.columns = [
    'Optimizer', 
    'Model Architecture', 
    'Dataset', 
    'Data Augmentation', 
    'Final Accuracy Mean (%)', 
    'Final Accuracy Std (%)',
    'Total Time Mean (s)', 
    'Total Time Std (s)', 
    'Total Time Mean (min)', 
    'Total Time Std (min)', 
]

# Display the DataFrame
print(grouped_df)

print(grouped_df.loc[grouped_df["Optimizer"] == "sgd"])