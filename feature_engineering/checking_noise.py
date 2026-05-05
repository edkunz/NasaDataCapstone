# Checking the noise_features.csv to view the corrosponding images based on filenamein visuals/boiling_plots folder
import pandas as pd
import matplotlib.pyplot as plt
# Load the noise features CSV
noise_features = pd.read_csv("data/noise_features.csv")
features = pd.read_csv("data/features.csv")

# Create non_noise_features by filtering out noise files
noise_files = set(noise_features["file_name"])
non_noise_features = features[~features["file_name"].isin(noise_files)]
out_path = "data/non_noise_features.csv"
non_noise_features.to_csv(out_path, index=False)
print(f"Non-noise features saved to '{out_path}'.")

# Display the noise correqponding images
for file in noise_features["file_name"]:
    img_path = f"visuals/boiling_plots/{file.replace('.csv', '.png')}"
    try:
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(f"Noise File: {file}")
        plt.axis('off')
        plt.show()
    except FileNotFoundError:
        print(f"Image for '{file}' not found at '{img_path}'.")