import os
import re
import numpy as np
import argparse

import re

def extract_test_accuracy(file_path, search_str):
    """
    Extract the first and last test accuracy from a log file.
    
    Args:
        file_path (str): Path to the log file.
        search_str (str): Search string to identify the accuracy type (e.g., 'accuracy').
    
    Returns:
        tuple: (first_test_accuracy, last_test_accuracy) as floats.
               If one of them is not found, it will return None for that value.
    """
    first_test_accuracy = None
    last_test_accuracy = None

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Iterate through lines to find the first and last matches
    for line in lines:
        match = re.search(rf"test {search_str}=([\d.]+)", line)
        if not match:
            match = re.search(rf"Test {search_str}=([\d.]+)", line)
        if match:
            accuracy = float(match.group(1))
            # First occurrence
            if first_test_accuracy is None:
                first_test_accuracy = accuracy
            # Last occurrence
            last_test_accuracy = accuracy  # Keep updating to get the last one
    
    return first_test_accuracy, last_test_accuracy


import os
import numpy as np

def calculate_statistics(folder_path, keystr, search_str):
    """
    Process all log files in a folder to calculate average and standard error of 
    both the first and last test accuracy.
    
    Args:
        folder_path (str): Path to the folder containing log files.
        keystr (str): Optional filter string for file names.
        search_str (str): Search string to identify the accuracy type (e.g., 'accuracy').
    
    Returns:
        dict: Statistics for first and last test accuracies:
              {
                  'first': {'avg': float, 'std': float, 'count': int},
                  'last': {'avg': float, 'std': float, 'count': int}
              }
    """
    first_test_accuracies = []
    last_test_accuracies = []
    
    # Process all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Filter files based on keystr and log file extension
        if keystr is not None and keystr not in file_name:
            continue
        if os.path.isfile(file_path) and file_name.endswith('.log'):  # Ensure only log files are processed
            # Extract first and last test accuracy from the file
            first_accuracy, last_accuracy = extract_test_accuracy(file_path, search_str)
            if first_accuracy is not None:
                first_test_accuracies.append(first_accuracy)
            if last_accuracy is not None:
                last_test_accuracies.append(last_accuracy)

    # Check if any valid accuracies were found
    if not first_test_accuracies and not last_test_accuracies:
        raise ValueError(f"No valid test accuracy values found in the folder: {folder_path}")

    # Calculate statistics for first and last accuracies
    stats = {}

    if first_test_accuracies:
        first_test_accuracies = np.array(first_test_accuracies)
        stats['first'] = {
            'avg': np.mean(first_test_accuracies),
            'std': np.std(first_test_accuracies, ddof=1),
            'max': np.max(first_test_accuracies),
            'count': len(first_test_accuracies),
        }
    else:
        stats['first'] = {'avg': None, 'std': None, 'max': None, 'count': 0}

    if last_test_accuracies:
        last_test_accuracies = np.array(last_test_accuracies)
        stats['last'] = {
            'avg': np.mean(last_test_accuracies),
            'std': np.std(last_test_accuracies, ddof=1),
            'max': np.max(last_test_accuracies),
            'count': len(last_test_accuracies),
        }
    else:
        stats['last'] = {'avg': None, 'std': None, 'max': None, 'count': 0}

    return stats


def process_all_subfolders(main_folder_path, keystr, search_str='accuracy'):
    """
    Process all subfolders in a main folder to compute statistics for each subfolder.
    Args:
        main_folder_path (str): Path to the main folder containing subfolders.
    Returns:
        dict: Mapping of subfolder name to (average accuracy, standard error).
    """
    results = {}

    # Iterate through each subfolder in the main folder
    for subfolder_name in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder_name)
        subfolder_path = os.path.join(subfolder_path, 'log')
        
        if os.path.isdir(subfolder_path):  # Check if it's a subfolder
            try:
                stats = calculate_statistics(subfolder_path, keystr, search_str)
                results[subfolder_name] = stats
            except ValueError as e:
                print(e)

    return results

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='../results/20241225_qenc_44iqp_full_train_qfm')
    parser.add_argument('--keystr', type=str, default=None)
    parser.add_argument('--search_str', type=str, default='accuracy')
    
    args = parser.parse_args()
    print(args)
    keystr = args.keystr
    search_str = args.search_str

    results = process_all_subfolders(args.folder, keystr=keystr, search_str=search_str)

    # Print results for each subfolder
    for subfolder, stats in results.items():
        print(f"Subfolder: {subfolder}")
        if search_str == 'accuracy':
            print(f"{stats['last']['count']} Test first accuracy = {stats['first']['avg']:.2f}%, std={stats['first']['std']:.2f}%, max={stats['first']['max']:.2f}%; ; last accuracy = {stats['last']['avg']:.2f}%, std={stats['last']['std']:.2f}%, max={stats['last']['max']:.2f}%")
        else:
            print(f"{stats['last']['count']} Test first loss = {stats['first']['avg']:.4f}, std={stats['first']['std']:.4f}, max={stats['first']['max']:.4f}; last loss = {stats['last']['avg']:.4f}, std={stats['last']['std']:.4f}, max={stats['last']['max']:.4f}")
        print()

