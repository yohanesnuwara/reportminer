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
