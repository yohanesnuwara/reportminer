import argparse
import os
import datetime
import io
import base64

from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportminer import rag, rag_folder

app = Flask(__name__)
app.static_folder = 'static'

# Global variable to store models
rag_models = None

def run_embedding(base_dir):
    global rag_models
    print('Setting up RAG models...')
    rag_models = rag.setup_model2()

    print('Normalizing folder structure...')
    destination_dir = base_dir
    rag_folder.normalize_folder_structure(base_dir, destination_dir)

    print('Embedding documents...')
    start_time = datetime.datetime.now()
    rag_models = rag_folder.Process(base_dir, rag_models)
    finish_time = datetime.datetime.now()
    print('Embedding completed in:', finish_time - start_time)

def chatbot_response(msg):
    res, source, image = rag_folder.Ask(msg, rag_models)
    document, page, score = source

    doc = rag_folder.retrieve_original_filepath(document)

    answer_text = f"""{res}

Source: {doc}
Page: {page}"""

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "answer": answer_text,
        "image": img_base64
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get("msg")
    response_data = chatbot_response(userText)
    return jsonify(response_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask chatbot with embedding.")
    parser.add_argument('-i', type=str, required=True, help='Folder path to embed')
    args = parser.parse_args()

    run_embedding(args.i)
    app.run(port=5000)
