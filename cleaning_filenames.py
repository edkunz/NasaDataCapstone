import os
import re

def extract_clean_filename(path_str):
    """
    Extracts the meaningful filename by removing path, 'MATLAB' prefix, and trailing image extensions like '.png'.
    Keeps the original '.csv' suffix if present.

    Examples:
        'Data/After_May/MATLAB 1-00 PM Fri, Jun 28, 2024 Run8 .csv' -> '1-00 PM Fri, Jun 28, 2024 Run8 .csv'
        '1-00 PM Fri, Jun 28, 2024 Run8 .csv' -> '1-00 PM Fri, Jun 28, 2024 Run8 .csv'
        'MATLAB 1-00 PM Fri, Jun 28, 2024 Run8 .csv.png' -> '1-00 PM Fri, Jun 28, 2024 Run8 .csv'
    """
    filename = os.path.basename(path_str)

    # Remove 'MATLAB' prefix if present
    if filename.startswith("MATLAB "):
        filename = filename[len("MATLAB "):]

    # Remove trailing .png or similar extensions if the true name ends in .csv
    match = re.search(r"(.*\.csv)\b", filename)
    if match:
        return match.group(1)

    # If no .csv match, return cleaned filename anyway
    return filename