import os
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import nest_asyncio
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

# PPT libraries
import spire.presentation
from spire.presentation import Presentation
from spire.presentation.common import *

# DOC libraries
import spire.doc
from spire.doc import Document
from spire.doc.common import *

# XLS libraries
from aspose.cells import Workbook, SaveFormat
from aspose.cells.rendering import ImageOrPrintOptions, SheetRender

import glob
import shutil
import pandas as pd
from PyPDF2 import PdfReader

def find_all_pdfs(root_directory, data_directory='./data'):
    """
    Collect all PDF reports in a folder
    The PDFs must be structured in a way like this:

    root_directory
    |_ sub directory 1
        |_ pdf 1
        |_ sub sub directory
            |_ pdf 2
    |_ sub directory 2
        |_ sub sub directory
            |_ pdf 3
            |_ pdf ...
    |_ ...

    It will collect all PDFs and order them into each of its sub directories

    Args:

    root_directory (str): The path of root directory
    data_directory (str): The path of data directory that is ordered by subdirectory. Default is './data'
    """
    # Create the 'data' directory if it doesn't exist
    os.makedirs(data_directory, exist_ok=True)

    # Process per well
    for well_path in glob.glob(root_directory + '/*'):
        # Extract the well name from the path
        well_name = os.path.basename(well_path)

        # Create a subdirectory for the well in the 'data' directory
        well_data_directory = os.path.join(data_directory, well_name)
        os.makedirs(well_data_directory, exist_ok=True)

        # Find all PDF files in the current well's directory and subdirectories
        pdf_files = glob.glob(f'{well_path}/**/*.pdf', recursive=True)

        # Copy each PDF file to the corresponding well's subdirectory
        for pdf_file in pdf_files:
            try:
                # Get the filename from the PDF path
                pdf_filename = os.path.basename(pdf_file)

                # Define the target path for the PDF
                target_path = os.path.join(well_data_directory, pdf_filename)

                # Copy the file
                shutil.copy(pdf_file, target_path)
                print(f"Copied: {pdf_file} -> {target_path}")
            except Exception as e:
                print(f"Error copying {pdf_file}: {e}")

    print("All PDF files have been copied.")

def count_pdf_pages_in_subdirectories(root_directory):
    """
    Counts the number of pages in each PDF file within each subdirectory of the root directory.

    Args:
        root_directory (str): The path to the root directory.

    Returns:
        pd.DataFrame: A DataFrame with subdirectory names, PDF file names, and page counts.
    """
    pdf_page_counts = []

    # Walk through the root directory
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.pdf'):  # Check if the file is a PDF
                pdf_path = os.path.join(subdir, file)
                try:
                    # Read the PDF and count the pages
                    reader = PdfReader(pdf_path)
                    num_pages = len(reader.pages)

                    # Get the relative subdirectory name
                    subdir_name = os.path.relpath(subdir, root_directory)

                    # Append the result to the list
                    pdf_page_counts.append({
                        "Subdirectory": subdir_name,
                        "PDF File": file,
                        "Page Count": num_pages
                    })
                except Exception as e:
                    print(f"Error reading {pdf_path}: {e}")

    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(pdf_page_counts)

    # Sort the DataFrame by subdirectory and file name for clarity
    df = df.sort_values(["Subdirectory", "PDF File"]).reset_index(drop=True)

    return df

# def Process(base_folder, models, dpi=100, index_name='rag'):
#     """
#     Processes files in a folder based on their format (.xls, .ppt, .pdf) and applies specific processing.

#     Args:
#         base_folder (str): The path to the folder containing the files.
#         models (list): List that consists of embedding model, VL model, and VL processor.
#         dpi (int): The resolution of the images for PDF files. Default is 100.
#         index_name (str): Index name for document retrieval.

#     Returns:
#         list: Updated models after processing files.
#     """
#     # Retrieve docs retrieval model as the first element of model input
#     docs_retrieval_model = models[0]

#     # Define output folder
#     output_base_folder = os.path.join(os.getcwd(), 'saved_images')
#     if not os.path.exists(output_base_folder):
#         os.makedirs(output_base_folder)

#     pdf_base_folder = os.path.join(os.getcwd(), 'pdf_copy')
#     if not os.path.exists(pdf_base_folder):
#         os.makedirs(pdf_base_folder)        

#     # ppt_base_folder = os.path.join(os.getcwd(), 'converted_ppt')
#     # if not os.path.exists(ppt_base_folder):
#     #     os.makedirs(ppt_base_folder)

#     # Get the list of files in the folder
#     files = os.listdir(base_folder)

#     # Process image files
#     img_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]
#     for img_file in img_files:
#         print('Processing Image:', img_file)
#         img_path = os.path.join(base_folder, img_file)
#         img_name = os.path.splitext(img_file)[0]        
#         sub_folder = os.path.join(output_base_folder, img_name)
#         if not os.path.exists(sub_folder):
#             os.makedirs(sub_folder)    

#         # Convert image to PDF
#         image = Image.open(img_path)
#         if image.mode in ("RGBA", "P"):
#             image = image.convert("RGB")
        
#         # Resize image
#         width, height = image.size
#         new_width, new_height = width // 2, height // 2
#         image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
#         # Save as PDF
#         pdf_output_path = os.path.join(pdf_base_folder, f"{img_name}.pdf")        
#         image_resized.save(pdf_output_path)

#         # Copy image to image folder
#         image_file_path = os.path.join(sub_folder, f"page_1.jpg")
#         shutil.copy(img_path, image_file_path)
      

#     # Process PDF files
#     pdf_files = [f for f in files if f.endswith(('.pdf', '.PDF'))]
#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(base_folder, pdf_file)
#         pdf_name = os.path.splitext(pdf_file)[0]
#         sub_folder = os.path.join(output_base_folder, pdf_name)
#         if not os.path.exists(sub_folder):
#             os.makedirs(sub_folder)

#         # Convert the PDF to images
#         images = convert_from_path(pdf_path, dpi=dpi)
#         for i, img in enumerate(images):
#             image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
#             img.save(image_file_path, "JPEG")
#         print(f"Images for '{pdf_file}' have been saved in folder: {sub_folder}")

#         # Copy the PDF file to the output folder
#         copied_pdf_path = os.path.join(pdf_base_folder, f"{pdf_file}")
#         shutil.copy(pdf_path, copied_pdf_path)

#     # Process Word files
#     doc_files = [f for f in files if f.endswith(('.doc', '.docx'))]
#     for doc_file in doc_files:
#         print('Processing Word:', doc_file)
#         doc_path = os.path.join(base_folder, doc_file)
#         doc_name = os.path.splitext(doc_file)[0]
#         sub_folder = os.path.join(output_base_folder, doc_name)
#         if not os.path.exists(sub_folder):
#             os.makedirs(sub_folder)

#         # Convert DOC to PDF
#         wordfile = Document()
#         wordfile.LoadFromFile(doc_path)

#         pdf_output_path = os.path.join(pdf_base_folder, f"{os.path.splitext(doc_file)[0]}.pdf")        


#         wordfile.SaveToFile(pdf_output_path, spire.doc.FileFormat.PDF)
#         wordfile.Dispose()

#         # Convert PDF to images
#         images = convert_from_path(pdf_output_path, dpi=dpi)
#         for i, img in enumerate(images):
#             image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
#             img.save(image_file_path, "JPEG")

#         print(f"Images for '{doc_file}' have been saved in folder: {sub_folder}")

#     # Process Presentation files
#     ppt_files = [f for f in files if f.endswith(('.ppt', '.pptx'))]
#     for ppt_file in ppt_files:
#         print('Processing PPT:', ppt_file)
#         ppt_path = os.path.join(base_folder, ppt_file)
#         ppt_name = os.path.splitext(ppt_file)[0]
#         sub_folder = os.path.join(output_base_folder, ppt_name)
#         if not os.path.exists(sub_folder):
#             os.makedirs(sub_folder)

#         # Convert PPT to PDF
#         presentation = Presentation()
#         presentation.LoadFromFile(ppt_path)

#         # pdf_output_path = os.path.join(ppt_base_folder, f"{os.path.splitext(ppt_file)[0]}.pdf")
#         pdf_output_path = os.path.join(pdf_base_folder, f"{os.path.splitext(ppt_file)[0]}.pdf")        


#         presentation.SaveToFile(pdf_output_path, spire.presentation.FileFormat.PDF)
#         presentation.Dispose()

#         # Convert PDF to images
#         images = convert_from_path(pdf_output_path, dpi=dpi)
#         for i, img in enumerate(images):
#             image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
#             img.save(image_file_path, "JPEG")

#         print(f"Images for '{ppt_file}' have been saved in folder: {sub_folder}")

#     # Process Excel files
#     xls_files = [f for f in files if f.endswith(('.xls', '.xlsx'))]
#     for xls_file in xls_files:
#         print('Processing XLS:', xls_file)
#         xls_path = os.path.join(base_folder, xls_file)
#         xls_name = os.path.splitext(xls_file)[0]
#         sub_folder = os.path.join(output_base_folder, xls_name)
#         if not os.path.exists(sub_folder):
#             os.makedirs(sub_folder)

#         # Load the Excel workbook using Aspose.Cells
#         workbook = Workbook(xls_path)     

#         # Convert the Excel workbook directly to PDF
#         pdf_output_path = os.path.join(pdf_base_folder, f"{xls_name}.pdf")
#         workbook.save(pdf_output_path, SaveFormat.PDF)   

#         # Convert PDF to images
#         images = convert_from_path(pdf_output_path, dpi=dpi)
#         for i, img in enumerate(images):
#             image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
#             img.save(image_file_path, "JPEG")

#         print(f"Images for '{xls_file}' have been saved in folder: {sub_folder}")

#         # # Iterate through worksheets
#         # for sheet_index, worksheet in enumerate(workbook.worksheets):
#         #     imgOptions = ImageOrPrintOptions()
#         #     imgOptions.save_format = SaveFormat.JPG  # Set the output format to JPG
#         #     imgOptions.horizontal_resolution = dpi  # Set horizontal DPI (higher value for better resolution)
#         #     imgOptions.vertical_resolution = dpi    # Set vertical DPI (higher value for better resolution)

#         #     # Create a SheetRender object for the worksheet
#         #     sheet_render = SheetRender(worksheet, imgOptions)

#         #     # Render each page of the worksheet
#         #     for page_number in range(sheet_render.page_count):
#         #         # output_filename = os.path.join(sub_folder, f"page_{sheet_index + 1}_page_{page_number + 1}.jpg")
#         #         output_filename = os.path.join(sub_folder, f"page_{sheet_index + 1}.jpg")
#         #         sheet_render.to_image(page_number, output_filename)

#         # print(f"Images for '{xls_file}' have been saved in folder: {sub_folder}")


#     # Index documents using the embedding model
#     docs_retrieval_model.index(
#         input_path=pdf_base_folder, ### can be base_folder also 
#         index_name=index_name,
#         store_collection_with_index=False,
#         overwrite=True
#     )

#     # Update models
#     models[0] = docs_retrieval_model
#     return models

# def Process(base_folder, models, dpi=100, index_name='rag'):
#     """
#     Processes files in a folder based on their format (.xls, .ppt, .pdf) and applies specific processing.

#     Args:
#         - base_folder (str): The path to the folder containing the files.

#         base_folder may have subfolder such as well name, and inside each there are subfolders

#         base_folder
#         \_well 1
#             \_ pdf1
#             \_subfolder 1.1
#                 \_ excel2
#                 \_ subfolder 1.1.1
#                     \_ jpg
#                 \_ ...
#         \_well 2
#             \_ ...
#         \_well n
#             \_ ...

#         Later the individual files will be re-structured to be put each well folder
            
#         - models (list): List that consists of embedding model, VL model, and VL processor.
#         - metadata (list of dict): List of dictionary of metadata. Can be well name or file path.
#         - dpi (int): The resolution of the images for PDF files. Default is 100.
#         - index_name (str): Index name for document retrieval.

#     Returns:
#         list: Updated models after processing files.
#     """
#     # 1 - Normalize file structure inside the base folder
#     # Base folder have subfolders and we need to put all the files directly inside the well folder
#     destination_folder = base_folder

#     # The result is dataframe consist of filename, source file path (original), and renamed file path (after restructured)
#     df = normalize_folder_structure(base_folder, destination_folder)
    
#     # 2 - Process files of different format
#     # Retrieve docs retrieval model as the first element of model input
#     docs_retrieval_model = models[0]

#     # Define output folder
#     output_base_folder = os.path.join(os.getcwd(), 'saved_images')
#     if not os.path.exists(output_base_folder):
#         os.makedirs(output_base_folder)

#     pdf_base_folder = os.path.join(os.getcwd(), 'pdf_copy')
#     if not os.path.exists(pdf_base_folder):
#         os.makedirs(pdf_base_folder)        


#     # Get the list of files in the folder
#     files = os.listdir(base_folder)

#     # 2a - Process image files
#     img_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]
#     for img_file in img_files:
#         print('Processing Image:', img_file)
#         img_path = os.path.join(base_folder, img_file)
#         img_name = os.path.splitext(img_file)[0]        
#         sub_folder = os.path.join(output_base_folder, img_name)
#         if not os.path.exists(sub_folder):
#             os.makedirs(sub_folder)    

#         # Convert image to PDF
#         image = Image.open(img_path)
#         if image.mode in ("RGBA", "P"):
#             image = image.convert("RGB")
        
#         # Resize image
#         width, height = image.size
#         new_width, new_height = width // 2, height // 2
#         image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
#         # Save as PDF
#         pdf_output_path = os.path.join(pdf_base_folder, f"{img_name}.pdf")        
#         image_resized.save(pdf_output_path)

#         # Copy image to image folder
#         image_file_path = os.path.join(sub_folder, f"page_1.jpg")
#         shutil.copy(img_path, image_file_path)
      

#     # 2b - Process PDF files
#     pdf_files = [f for f in files if f.endswith(('.pdf', '.PDF'))]
#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(base_folder, pdf_file)
#         pdf_name = os.path.splitext(pdf_file)[0]
#         sub_folder = os.path.join(output_base_folder, pdf_name)
#         if not os.path.exists(sub_folder):
#             os.makedirs(sub_folder)

#         # Convert the PDF to images
#         images = convert_from_path(pdf_path, dpi=dpi)
#         for i, img in enumerate(images):
#             image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
#             img.save(image_file_path, "JPEG")
#         print(f"Images for '{pdf_file}' have been saved in folder: {sub_folder}")

#         # Copy the PDF file to the output folder
#         copied_pdf_path = os.path.join(pdf_base_folder, f"{pdf_file}")
#         shutil.copy(pdf_path, copied_pdf_path)

#     # 2c - Process Word files
#     doc_files = [f for f in files if f.endswith(('.doc', '.docx'))]
#     for doc_file in doc_files:
#         print('Processing Word:', doc_file)
#         doc_path = os.path.join(base_folder, doc_file)
#         doc_name = os.path.splitext(doc_file)[0]
#         sub_folder = os.path.join(output_base_folder, doc_name)
#         if not os.path.exists(sub_folder):
#             os.makedirs(sub_folder)

#         # Convert DOC to PDF
#         wordfile = Document()
#         wordfile.LoadFromFile(doc_path)

#         pdf_output_path = os.path.join(pdf_base_folder, f"{os.path.splitext(doc_file)[0]}.pdf")        


#         wordfile.SaveToFile(pdf_output_path, spire.doc.FileFormat.PDF)
#         wordfile.Dispose()

#         # Convert PDF to images
#         images = convert_from_path(pdf_output_path, dpi=dpi)
#         for i, img in enumerate(images):
#             image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
#             img.save(image_file_path, "JPEG")

#         print(f"Images for '{doc_file}' have been saved in folder: {sub_folder}")

#     # 2d - Process Presentation files
#     ppt_files = [f for f in files if f.endswith(('.ppt', '.pptx'))]
#     for ppt_file in ppt_files:
#         print('Processing PPT:', ppt_file)
#         ppt_path = os.path.join(base_folder, ppt_file)
#         ppt_name = os.path.splitext(ppt_file)[0]
#         sub_folder = os.path.join(output_base_folder, ppt_name)
#         if not os.path.exists(sub_folder):
#             os.makedirs(sub_folder)

#         # Convert PPT to PDF
#         presentation = Presentation()
#         presentation.LoadFromFile(ppt_path)

#         # pdf_output_path = os.path.join(ppt_base_folder, f"{os.path.splitext(ppt_file)[0]}.pdf")
#         pdf_output_path = os.path.join(pdf_base_folder, f"{os.path.splitext(ppt_file)[0]}.pdf")        


#         presentation.SaveToFile(pdf_output_path, spire.presentation.FileFormat.PDF)
#         presentation.Dispose()

#         # Convert PDF to images
#         images = convert_from_path(pdf_output_path, dpi=dpi)
#         for i, img in enumerate(images):
#             image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
#             img.save(image_file_path, "JPEG")

#         print(f"Images for '{ppt_file}' have been saved in folder: {sub_folder}")

#     # 2e - Process Excel files
#     xls_files = [f for f in files if f.endswith(('.xls', '.xlsx'))]
#     for xls_file in xls_files:
#         print('Processing XLS:', xls_file)
#         xls_path = os.path.join(base_folder, xls_file)
#         xls_name = os.path.splitext(xls_file)[0]
#         sub_folder = os.path.join(output_base_folder, xls_name)
#         if not os.path.exists(sub_folder):
#             os.makedirs(sub_folder)

#         # Load the Excel workbook using Aspose.Cells
#         workbook = Workbook(xls_path)     

#         # Convert the Excel workbook directly to PDF
#         pdf_output_path = os.path.join(pdf_base_folder, f"{xls_name}.pdf")
#         workbook.save(pdf_output_path, SaveFormat.PDF)   

#         # Convert PDF to images
#         images = convert_from_path(pdf_output_path, dpi=dpi)
#         for i, img in enumerate(images):
#             image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
#             img.save(image_file_path, "JPEG")

#         print(f"Images for '{xls_file}' have been saved in folder: {sub_folder}")

#     # 3 - Create file metadata for embedding model

#     # Use glob to retrieve the files SEQUENTIALLY
#     # Collect all matching PDF files into a list
#     files = []
#     for f in glob.glob(pdf_base_folder + '/*'):
#         filename = f.split('/')[-1].split('.')[0]
#         files.append(filename)

#     # Create metadata dictionary from the globbed filepath
#     metadata = [{"path": source_path} for source_path in files]    
     

#     # 4 - Index documents using the embedding model
#     print('Indexing Documents with ColPali ...')
#     docs_retrieval_model.index(
#         input_path=pdf_base_folder, ### can be base_folder also 
#         index_name=index_name,
#         store_collection_with_index=False,
#         metadata=metadata, 
#         overwrite=True
#     )

#     # Update models
#     models[0] = docs_retrieval_model
#     return models

def Process(base_folder, models, dpi=100, index_name='rag'):
    """
    Processes files in a folder based on their format (.xls, .ppt, .pdf) and applies specific processing.

    Args:
        - base_folder (str): The path to the folder containing the files.

        base_folder may have subfolder such as well name, and inside each there are subfolders

        base_folder
        \_well 1
            \_ pdf1
            \_subfolder 1.1
                \_ excel2
                \_ subfolder 1.1.1
                    \_ jpg
                \_ ...
        \_well 2
            \_ ...
        \_well n
            \_ ...

        Later the individual files will be re-structured to be put each well folder
            
        - models (list): List that consists of embedding model, VL model, and VL processor.
        - metadata (list of dict): List of dictionary of metadata. Can be well name or file path.
        - dpi (int): The resolution of the images for PDF files. Default is 100.
        - index_name (str): Index name for document retrieval.

    Returns:
        list: Updated models after processing files.
    """
    # 1 - Normalize file structure inside the base folder
    # Base folder have subfolders and we need to put all the files directly inside the well folder
    destination_folder = base_folder

    # The result is dataframe consist of filename, source file path (original), and renamed file path (after restructured)
    df = normalize_folder_structure(base_folder, destination_folder)
    
    # 2 - Process files of different format
    # Retrieve docs retrieval model as the first element of model input
    docs_retrieval_model = models[0]

    # Define output folder
    output_base_folder = os.path.join(os.getcwd(), 'saved_images')
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    pdf_base_folder = os.path.join(os.getcwd(), 'pdf_copy')
    if not os.path.exists(pdf_base_folder):
        os.makedirs(pdf_base_folder)        


    # Get the list of files in the folder
    files = os.listdir(base_folder)

    # 2a - Process image files
    img_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]
    for img_file in img_files:
        print('Processing Image:', img_file)
        img_path = os.path.join(base_folder, img_file)
        # img_name = os.path.splitext(img_file)[0]   
        img_name = img_file
        sub_folder = os.path.join(output_base_folder, img_name)
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)    

        # Convert image to PDF
        image = Image.open(img_path)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # Resize image
        width, height = image.size
        new_width, new_height = width // 2, height // 2
        image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save as PDF
        pdf_output_path = os.path.join(pdf_base_folder, f"{img_name}.pdf")        
        image_resized.save(pdf_output_path)

        # Copy image to image folder
        image_file_path = os.path.join(sub_folder, f"page_1.jpg")
        shutil.copy(img_path, image_file_path)
      

    # 2b - Process PDF files
    pdf_files = [f for f in files if f.endswith(('.pdf', '.PDF'))]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(base_folder, pdf_file)
        # pdf_name = os.path.splitext(pdf_file)[0]
        pdf_name = pdf_file
        sub_folder = os.path.join(output_base_folder, pdf_name)
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        # Convert the PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)
        for i, img in enumerate(images):
            image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
            img.save(image_file_path, "JPEG")
        print(f"Images for '{pdf_file}' have been saved in folder: {pdf_name}")

        # Copy the PDF file to the output folder
        copied_pdf_path = os.path.join(pdf_base_folder, f"{pdf_name}.pdf")
        shutil.copy(pdf_path, copied_pdf_path)

    # 2c - Process Word files
    doc_files = [f for f in files if f.endswith(('.doc', '.docx'))]
    for doc_file in doc_files:
        print('Processing Word:', doc_file)
        doc_path = os.path.join(base_folder, doc_file)
        # doc_name = os.path.splitext(doc_file)[0]
        doc_name = doc_file
        sub_folder = os.path.join(output_base_folder, doc_name)
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        # Convert DOC to PDF
        wordfile = Document()
        wordfile.LoadFromFile(doc_path)

        pdf_output_path = os.path.join(pdf_base_folder, f"{doc_name}.pdf")        


        wordfile.SaveToFile(pdf_output_path, spire.doc.FileFormat.PDF)
        wordfile.Dispose()

        # Convert PDF to images
        images = convert_from_path(pdf_output_path, dpi=dpi)
        for i, img in enumerate(images):
            image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
            img.save(image_file_path, "JPEG")

        print(f"Images for '{doc_file}' have been saved in folder: {sub_folder}")

    # # 2d - Process Presentation files
    # ppt_files = [f for f in files if f.endswith(('.ppt', '.pptx'))]
    # for ppt_file in ppt_files:
    #     print('Processing PPT:', ppt_file)
    #     ppt_path = os.path.join(base_folder, ppt_file)
    #     # ppt_name = os.path.splitext(ppt_file)[0]
    #     ppt_name = ppt_file
    #     sub_folder = os.path.join(output_base_folder, ppt_name)
    #     if not os.path.exists(sub_folder):
    #         os.makedirs(sub_folder)

    #     # Convert PPT to PDF
    #     presentation = Presentation()
    #     presentation.LoadFromFile(ppt_path)

    #     # pdf_output_path = os.path.join(ppt_base_folder, f"{os.path.splitext(ppt_file)[0]}.pdf")
    #     pdf_output_path = os.path.join(pdf_base_folder, f"{ppt_name}.pdf")        


    #     presentation.SaveToFile(pdf_output_path, spire.presentation.FileFormat.PDF)
    #     presentation.Dispose()

    #     # Convert PDF to images
    #     images = convert_from_path(pdf_output_path, dpi=dpi)
    #     for i, img in enumerate(images):
    #         image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
    #         img.save(image_file_path, "JPEG")

    #     print(f"Images for '{ppt_file}' have been saved in folder: {sub_folder}")

    # 2e - Process Excel files
    xls_files = [f for f in files if f.endswith(('.xls', '.xlsx'))]
    for xls_file in xls_files:
        print('Processing XLS:', xls_file)
        xls_path = os.path.join(base_folder, xls_file)
        # xls_name = os.path.splitext(xls_file)[0]
        xls_name = xls_file
        sub_folder = os.path.join(output_base_folder, xls_name)
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        # Load the Excel workbook using Aspose.Cells
        workbook = Workbook(xls_path)     

        # Convert the Excel workbook directly to PDF
        pdf_output_path = os.path.join(pdf_base_folder, f"{xls_name}.pdf")
        workbook.save(pdf_output_path, SaveFormat.PDF)   

        # Convert PDF to images
        images = convert_from_path(pdf_output_path, dpi=dpi)
        for i, img in enumerate(images):
            image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
            img.save(image_file_path, "JPEG")

        print(f"Images for '{xls_file}' have been saved in folder: {sub_folder}")

    # 3 - Create file metadata for embedding model

    # Use glob to retrieve the files SEQUENTIALLY
    # Collect all matching PDF files into a list
    files = []
    for f in glob.glob(pdf_base_folder + '/*'):
        filename = f.split('/')[-1].split('.')[0]
        files.append(filename)

    # Create metadata dictionary from the globbed filepath
    metadata = [{"path": source_path} for source_path in files]    
     

    # 4 - Index documents using the embedding model
    print('Indexing Documents with ColPali ...')
    docs_retrieval_model.index(
        input_path=pdf_base_folder, ### can be base_folder also 
        index_name=index_name,
        store_collection_with_index=False,
        metadata=metadata, 
        overwrite=True
    )

    # Update models
    models[0] = docs_retrieval_model
    return models


def PDF_folder(pdf_folder, models, dpi=200, index_name='rag'):
    """
    Converts PDF files into images and stores them in sub-folders named after the PDF files.

    Args:
        pdf_folder (str): The path to the folder that contains PDF files.
        models (list): List that consists embedding model, VL model, and VL processor.
        dpi (int): The resolution of the images in DPI. Default is 200.
    """
    # Retrieve docs retrieval model as the first element of model input
    docs_retrieval_model = models[0]

    # Get the list of PDF files in the folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        # Path to the current PDF file
        pdf_path = os.path.join(pdf_folder, pdf_file)

        # Sub-folder named after the PDF file (without extension)
        pdf_name = os.path.splitext(pdf_file)[0]

        # Save images to folder
        output_base_folder = os.getcwd() + '/saved_images'
        sub_folder = os.path.join(output_base_folder, pdf_name)

        # Create the sub-folder if it doesn't exist
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        # Convert the PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)

        # Save images to the sub-folder
        for i, img in enumerate(images):
            image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
            img.save(image_file_path, "JPEG")

        print(f"Images for '{pdf_file}' have been saved in folder: {sub_folder}")

    # using embedding model to convert document to embedding
    docs_retrieval_model.index(
        input_path=pdf_folder,
        index_name=index_name,
        store_collection_with_index=False,
        overwrite=True
    )

    # Update models
    models[0] = docs_retrieval_model
    return models

def retrieve_image(filename, page_num):
    """
    Retrieve page image based on page number.

    Args:
        filename (str): The name of the file.
        page_num (int): Page number.

    """
    # Extract filename
    # filename = filename.split('/')[-1].split('.')[0]
    filename = filename.split('/')[-1].split('.')[:-1]
    filename = '.'.join(filename)    

    output_base_folder = os.path.join(os.getcwd(), 'saved_images', filename)
    image_path = os.path.join(output_base_folder, f"page_{page_num}.jpg")
    image = Image.open(image_path)
    return image

def RAG(text_query, models, k=1):
    """
    Retrieve k relevant pages using embedding model.

    Args:
        text_query (str): Query for the document. Can be in a form of question.
        models (list): List that consists embedding model, VL model, and VL processor.
        k (int): The number of relevant pages to retrieve. Default is 1.

    """
    # Retrieve docs retrieval model as the first element of model input
    docs_retrieval_model = models[0]

    # Read mapping of document ID to filename
    doc_mapping = docs_retrieval_model.get_doc_ids_to_file_names()

    # Run similarity search
    results = docs_retrieval_model.search(text_query, k=k)

    for result in results:
        doc_id = result['doc_id']
        page_num = result['page_num']
        score = result['score']
        filename = doc_mapping[doc_id]
        image = np.array(retrieve_image(filename, page_num))

        # Visualize page
        print(f"Report Name {filename}, Page {page_num} with Relevancy score {score}")
        plt.figure(figsize=(15, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        print('\n')

def Ask(text_query, models):
    """
    Perform question and answering task to document.

    Args:
        text_query (str): Query for the document. Can be in a form of question.
        models (list): List that consists embedding model, VL model, and VL processor.

    Return:
        Answer given from the VL model
    """
    # Retrieve docs retrieval model as the first element of model input
    docs_retrieval_model = models[0]

    # Read mapping of document ID to filename
    doc_mapping = docs_retrieval_model.get_doc_ids_to_file_names()

    # Run similarity search
    results = docs_retrieval_model.search(text_query, k=1)

    result = results[0]
    doc_id = result['doc_id']
    page_num = result['page_num']
    score = result['score']
    filename = doc_mapping[doc_id]
    image = retrieve_image(filename, page_num)


    # VL model and processor
    model = models[1]
    processor = models[2]

    chat_template = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Format your answer into markdown."},
            ],
        },        
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": text_query},
            ],
        }
    ]

    text = processor.apply_chat_template(chat_template, add_generation_prompt=True)

    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=5000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Source of information
    source = [filename, page_num, score]

    return output_text[0], source, image

def Ask_iterative(text_query, models, k=3):
    """
    Perform question and answering task to document. Sourcing more than one document

    Args:
        text_query (str): Query for the document. Can be in a form of question.
        models (list): List that consists embedding model, VL model, and VL processor.

    Return:
        Answer given from the VL model
    """
    # Retrieve docs retrieval model as the first element of model input
    docs_retrieval_model = models[0]

    # Read mapping of document ID to filename
    doc_mapping = docs_retrieval_model.get_doc_ids_to_file_names()

    # Run similarity search
    results = docs_retrieval_model.search(text_query, k=k)
    
    outputs = []
    sources = []
    images = []
    
    for i in range(len(results)):
        result = results[i]
        doc_id = result['doc_id']
        page_num = result['page_num']
        score = result['score']
        filename = doc_mapping[doc_id]
        image = retrieve_image(filename, page_num)


        # VL model and processor
        model = models[1]
        processor = models[2]

        chat_template = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Format your answer into markdown."},
                ],
            },        
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": text_query},
                ],
            }
        ]

        text = processor.apply_chat_template(chat_template, add_generation_prompt=True)

        inputs = processor(
            text=text,
            images=[image],
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=5000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        output_text = output_text[0]
        print(output_text)

        # Source of information
        source = [filename, page_num, score]

        outputs.append(output_text)
        sources.append(source)
        images.append(image)

    return outputs, sources, images

def Ask2(text_query, models):
    """
    Perform question and answering task to document. Specifically for DeepSeek and other models hosted in LMDeploy.

    Args:
        text_query (str): Query for the document. Can be in a form of question.
        models (list): List that consists embedding model, VL model, and VL processor.

    Return:
        Answer given from the VL model
    """
    # Retrieve docs retrieval model as the first element of model input
    docs_retrieval_model = models[0]

    # Read mapping of document ID to filename
    doc_mapping = docs_retrieval_model.get_doc_ids_to_file_names()

    # Run similarity search
    results = docs_retrieval_model.search(text_query, k=1)

    result = results[0]
    doc_id = result['doc_id']
    page_num = result['page_num']
    score = result['score']
    filename = doc_mapping[doc_id]
    image = retrieve_image(filename, page_num)


    # VL model and processor
    model = models[1]

    text_query = text_query + ' Answer briefly but explain more.'
    output_text = model((text_query, image))

    # Source of information
    source = [filename, page_num, score]

    return output_text.text, source, image

def normalize_folder_structure(base_dir, destination_dir):
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    rename_records = []

    for file_path in glob.glob(base_dir + "/**", recursive=True):
        if os.path.isfile(file_path):
            # Replace backslashes in full path with underscores
            normalized_name = file_path.replace("\\", "__").split("/")[-1]

            # Get parent folder and replace its backslashes too
            parent_path = os.path.dirname(file_path).replace("\\", "__")
            new_path = os.path.join(destination_dir, normalized_name)

            # Move the file
            shutil.move(file_path, new_path)

            print(f"Renamed: {file_path} -> {new_path}")

            rename_records.append({
                "source_file_path": file_path,
                "renamed_file_path": new_path
            })

    print("Renaming completed.")

    # Create a DataFrame for logging
    df = pd.DataFrame(rename_records, columns=["source_file_path", "renamed_file_path"])
    df['filename'] = df['renamed_file_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    return df

def retrieve_original_filepath(embedded_pdf_filepath, folder_delimitter='__'):
    """
    Retrieve original filepath from embedded PDF filepath

    GeoDataX filepath (or path metadata) is like this: ./pdf_copy/Pharos-1__Geophysics__seismic.pdf
    The filepath has delimitter '__' that indicates the folder delimitter '/'
    The task of this function to translate to original filepath: Pharos-1/Geophysics/seismic.pdf

    Arguments:

    embedded_pdf_filepath (path-like): Path to embedded PDF file 
    folder_delimitter (char): Delimitter of filepath. Default: '__'

    Output:

    original_filepath (path-like): Original filepath in the local
    """
    original_filepath = '/'.join(os.path.splitext(embedded_pdf_filepath)[0].split('/')[-1].split(folder_delimitter))
    return original_filepath
