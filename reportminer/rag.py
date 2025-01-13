import os
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def PDF_folder(pdf_folder, models, dpi=200):
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

def process_PDF(pdf_file, models, dpi=200, index_name='rag'):
    """
    Converts a single PDF file into images and saves them to a specified folder.

    Args:
        pdf_file (str): The path to the PDF file.
        models (list): List that consists embedding model, VL model, and VL processor.
        dpi (int): The resolution of the images in DPI. Default is 200.

    """
    # Retrieve docs retrieval model as the first element of model input
    docs_retrieval_model = models[0]

    # Convert the PDF to images
    images = convert_from_path(pdf_file, dpi=dpi)

    # Determine output folder
    output_base_folder = os.getcwd()  # Default to current working directory

    # Create a sub-folder named after the PDF file (without extension)
    pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
    output_folder = os.path.join(output_base_folder, 'saved_images')

    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each image to the folder
    for i, img in enumerate(images):
        file_path = os.path.join(output_folder, f"page_{i + 1}.jpg")
        img.save(file_path, "JPEG")

    print(f"Images for '{pdf_file}' have been saved in folder: {output_folder}")

    # using embedding model to convert document to embedding
    docs_retrieval_model.index(
        input_path=pdf_file,
        index_name=index_name,
        store_collection_with_index=False,
        overwrite=True
    )

    # Update models
    models[0] = docs_retrieval_model
    return models

def retrieve_image(page_num):
    """
    Retrieve page image based on page number.

    Args:
        page_num (int): Page number.

    """  
    output_base_folder = os.getcwd() + '/saved_images'
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

    # Run similarity search
    results = docs_retrieval_model.search(text_query, k=k)

    for result in results:
        doc_id = result['doc_id']
        page_num = result['page_num']
        score = result['score']
        image = np.array(retrieve_image(page_num))

        # Visualize page
        print(f"Page {page_num} with Relevancy score {score}")
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
    # Embedding model
    docs_retrieval_model = models[0]

    results = docs_retrieval_model.search(text_query, k=1)
    result = results[0]
    doc_id = result['doc_id']
    page_num = result['page_num']
    score = result['score']
    image = np.array(retrieve_image(page_num))


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
    return output_text[0]

def setup_model(embedding='vidore/colpali-v1.2', vlm='HuggingFaceTB/SmolVLM-Instruct'):
    """
    Install embedding model and visual language model

    Args:
        embedding (str): Model name of the embedding model from HuggingFace or Byaldi. Default is 'vidore/colpali-v1.2'.
        vlm (str): Model name of the visual language model from HuggingFace. Default is 'HuggingFaceTB/SmolVLM-Instruct'.
    """
    # Print selected model
    print('Selected embedding model:', embedding)
    print('Selected visual language model:', vlm)

    # Setup embedding model
    docs_retrieval_model = RAGMultiModalModel.from_pretrained(embedding)

    # Setup visual language model
    vl_model = Idefics3ForConditionalGeneration.from_pretrained(
        vlm,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    )
    vl_model.eval()

    # Setup VLM processor
    vl_processor = AutoProcessor.from_pretrained(vlm)    

    return [docs_retrieval_model, vl_model, vl_processor]


def load_model_from_pretrained(pretrained_embedding_index_name, vlm='HuggingFaceTB/SmolVLM-Instruct'):
    """
    Load pretrained embedding model indexes and setup VLM 

    Pretrained index must be inside ./.byaldi/<index-name>/...

    Args:
        pretrained_embedding_index_name (str, path like): Index name of pretrained embedding model.
        vlm (str): Model name of the visual language model from HuggingFace. Default is 'HuggingFaceTB/SmolVLM-Instruct'.
    """
    # Print selected model
    print('Selected embedding model - PRETRAINED:', pretrained_embedding_index_name)
    print('Selected visual language model:', vlm)

    # Setup embedding model
    docs_retrieval_model = RAGMultiModalModel.from_index(pretrained_embedding_index_name)

    # Setup visual language model
    vl_model = Idefics3ForConditionalGeneration.from_pretrained(
        vlm,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    )
    vl_model.eval()

    # Setup VLM processor
    vl_processor = AutoProcessor.from_pretrained(vlm)    

    return [docs_retrieval_model, vl_model, vl_processor]
