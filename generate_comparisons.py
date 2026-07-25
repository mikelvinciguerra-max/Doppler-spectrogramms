import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# --- Parameters to configure ---
# Folder where your generated folders are located (use "./" if the script is in the same directory)
parent_folder = "./" 

# Name of the output folder for the generated grids
output_folder = "comparison"

# Number of columns for the grid in the final large image
columns = 3 
# -------------------------------

print("Searching for folders...")
search_path = os.path.join(parent_folder, "plots_Tc_*")
folders = sorted(glob.glob(search_path))

if not folders:
    print("Error: No 'plots_Tc_...' folders found.")
    exit()

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
print(f"Output folder '{output_folder}' is ready.")

print("Gathering all unique filenames...")
unique_filenames = set()

# Scan all folders to find every unique .png file
for folder in folders:
    png_files = glob.glob(os.path.join(folder, "*.png"))
    for f in png_files:
        unique_filenames.add(os.path.basename(f))

if not unique_filenames:
    print("Error: No .png files found in the generated folders.")
    exit()

# Sort the filenames alphabetically for logical processing
unique_filenames = sorted(list(unique_filenames))
total_files = len(unique_filenames)
print(f"Found {total_files} unique images. Generating comparison grids...")

# Loop through every unique file found
for index, target_filename in enumerate(unique_filenames, start=1):
    print(f"Processing {index}/{total_files}: {target_filename}...")
    
    valid_folders = []
    for folder in folders:
        image_path = os.path.join(folder, target_filename)
        if os.path.exists(image_path):
            valid_folders.append(folder)
    
    num_images = len(valid_folders)
    rows = math.ceil(num_images / columns)
    
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 7, rows * 6))
    
    if num_images > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    for i, folder in enumerate(valid_folders):
        folder_name = os.path.basename(folder)
        parameters_title = folder_name.replace("plots_", "").replace("_", " | ")
        
        image_path = os.path.join(folder, target_filename)
        img = mpimg.imread(image_path)
        
        axes[i].imshow(img)
        axes[i].axis('off') 
        axes[i].set_title(parameters_title, fontsize=12, fontweight='bold', pad=10)
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    
    # Save the output file INSIDE the 'comparison' folder
    output_filename = os.path.join(output_folder, f"parameter_comparison_{target_filename.split('.')[0]}.jpg")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    # CRITICAL: Close the figure to free up memory before the next loop iteration
    plt.close(fig) 

print(f"\nSuccess! All comparison grids have been generated in the '{output_folder}' directory.")