"""
Run Embedding on selected file

Copyright ReportMiner 2015

Note: Run this in the reportminer folder. cd reportminer
"""
import argparse
import os 
import glob
import shutil
import datetime
from reportminer import rag, rag_folder

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process and embed documents.")
    parser.add_argument('-i', type=str, required=True,
                        help='Path of the file or directory to process')
    args = parser.parse_args()

    # Use the provided path as the base directory
    base_dir = args.i

    # Setup models ColPali and SMOL
    rag_models = rag.setup_model2()

    # Rename files uploaded using Runpodctl method
    destination_dir = base_dir 
    rag_folder.normalize_folder_structure(base_dir, destination_dir)
    # os.makedirs(destination_dir, exist_ok=True)

    # # Process all files recursively
    # for file_path in glob.glob(os.path.join(base_dir, "**"), recursive=True):
    #     if os.path.isfile(file_path):
    #         # Normalize the filename and extract just the name
    #         filename = os.path.basename(file_path.replace("\\", "/"))
    #         new_path = os.path.join(destination_dir, filename)
    #         shutil.move(file_path, new_path)
    #         print(f"Renamed: {file_path} -> {new_path}")
    # print("Renaming completed.")

    # Embed documents
    start_time = datetime.datetime.now()
    rag_models = rag_folder.Process(base_dir, rag_models)
    finish_time = datetime.datetime.now()

    print('Embedding completed in:', finish_time-start_time)  

if __name__ == "__main__":
    main()
