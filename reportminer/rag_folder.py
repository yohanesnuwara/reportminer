import os
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

import os
from pdf2image import convert_from_path

def Process(base_folder, models, dpi=100, index_name='rag'):
    """
    Processes files in a folder based on their format (.xls, .ppt, .pdf) and applies specific processing.
    """
    output_base_folder = os.path.join(os.getcwd(), 'saved_images')
    os.makedirs(output_base_folder, exist_ok=True)

    ppt_base_folder = os.path.join(os.getcwd(), 'converted_ppt')
    os.makedirs(ppt_base_folder, exist_ok=True)

    files = os.listdir(base_folder)

    # -------------------------------------------------------
    # 1. Check for file types
    # -------------------------------------------------------
    ppt_files = [f for f in files if f.lower().endswith(('.ppt', '.pptx'))]
    xls_files = [f for f in files if f.lower().endswith(('.xls', '.xlsx'))]

    # Flags to see if user actually has PPT or Excel files to process
    ppt_found = len(ppt_files) > 0
    xls_found = len(xls_files) > 0

    # Flags to check if required libs are missing
    missing_ppt_deps = False
    missing_xls_deps = False

    # -------------------------------------------------------
    # 2. If user has PPT files, try importing PPT dependencies
    # -------------------------------------------------------
    if ppt_found:
        try:
            from spire.presentation import *
            from spire.presentation.common import *
        except ImportError:
            missing_ppt_deps = True

    # -------------------------------------------------------
    # 3. If user has Excel files, try importing XLS dependencies
    # -------------------------------------------------------
    if xls_found:
        try:
            from aspose.cells import Workbook, SaveFormat
            from aspose.cells.rendering import ImageOrPrintOptions, SheetRender
        except ImportError:
            missing_xls_deps = True

    # -------------------------------------------------------
    # 4. Raise combined error if both missing
    # -------------------------------------------------------
    if missing_ppt_deps and missing_xls_deps:
        raise ImportError(
            "Both Excel and PowerPoint files are detected, but their dependencies are missing.\n"
            "Install both extras:  pip install \"reportminer[xls,ppt]\"\n"
            "Or install all:       pip install \"reportminer[all]\""
        )
    elif missing_ppt_deps:
        raise ImportError(
            "PowerPoint files are detected, but PPT dependencies are missing.\n"
            "Please install: pip install \"reportminer[ppt]\""
        )
    elif missing_xls_deps:
        raise ImportError(
            "Excel files are detected, but XLS dependencies are missing.\n"
            "Please install: pip install \"reportminer[xls]\""
        )

    # -------------------------------------------------------
    # 5. Process PDF files
    # -------------------------------------------------------
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(base_folder, pdf_file)
        pdf_name = os.path.splitext(pdf_file)[0]
        sub_folder = os.path.join(output_base_folder, pdf_name)
        os.makedirs(sub_folder, exist_ok=True)

        images = convert_from_path(pdf_path, dpi=dpi)
        for i, img in enumerate(images):
            image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
            img.save(image_file_path, "JPEG")
        print(f"PDF -> Images for '{pdf_file}' saved in folder: {sub_folder}")

    # -------------------------------------------------------
    # 6. Process PPT files
    # -------------------------------------------------------
    # By this point, if we got here, the spire.presentation library is available if needed.
    if ppt_found and not missing_ppt_deps:
        from spire.presentation import Presentation, FileFormat
        for ppt_file in ppt_files:
            ppt_path = os.path.join(base_folder, ppt_file)
            ppt_name = os.path.splitext(ppt_file)[0]
            sub_folder = os.path.join(output_base_folder, ppt_name)
            os.makedirs(sub_folder, exist_ok=True)

            pres = Presentation()
            pres.LoadFromFile(ppt_path)

            pdf_output_path = os.path.join(ppt_base_folder, f"{ppt_name}.pdf")
            pres.SaveToFile(pdf_output_path, FileFormat.PDF)
            pres.Dispose()

            images = convert_from_path(pdf_output_path, dpi=dpi)
            for i, img in enumerate(images):
                image_file_path = os.path.join(sub_folder, f"page_{i + 1}.jpg")
                img.save(image_file_path, "JPEG")
            print(f"PPT -> Images for '{ppt_file}' saved in folder: {sub_folder}")

    # -------------------------------------------------------
    # 7. Process Excel files
    # -------------------------------------------------------
    # Similarly, the Aspose.Cells import is valid here if needed.
    if xls_found and not missing_xls_deps:
        from asposecells.api import Workbook, ImageOrPrintOptions, SheetRender, SaveFormat
        for xls_file in xls_files:
            xls_path = os.path.join(base_folder, xls_file)
            xls_name = os.path.splitext(xls_file)[0]
            sub_folder = os.path.join(output_base_folder, xls_name)
            os.makedirs(sub_folder, exist_ok=True)

            workbook = Workbook(xls_path)
            for sheet_index, worksheet in enumerate(workbook.worksheets):
                img_options = ImageOrPrintOptions()
                img_options.save_format = SaveFormat.JPG
                img_options.horizontal_resolution = dpi
                img_options.vertical_resolution = dpi

                sheet_render = SheetRender(worksheet, img_options)
                for page_number in range(sheet_render.page_count):
                    output_filename = os.path.join(
                        sub_folder, f"sheet_{sheet_index+1}_page_{page_number+1}.jpg"
                    )
                    sheet_render.to_image(page_number, output_filename)

            print(f"XLS -> Images for '{xls_file}' saved in folder: {sub_folder}")

    # -------------------------------------------------------
    # 8. Index Documents (example)
    # -------------------------------------------------------
    docs_retrieval_model = models[0]
    docs_retrieval_model.index(
        input_path=base_folder,
        index_name=index_name,
        store_collection_with_index=False,
        overwrite=True
    )
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
    filename = filename.split('/')[-1].split('.')[0]

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
