import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for generating PNG image
import io
import base64
import numpy as np

from reportminer import rag, rag_folder

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.static_folder = 'static'

print('Loading pretrained embedding ...')
model = rag.load_model_from_pretrained('/home/reportminer/.byaldi/rag')

def chatbot_response(msg):
    # Ask the message
    res, source, image = rag_folder.Ask(msg, model)
    document, page, score = source

    # Retrieve filepath
    doc = rag_folder.retrieve_original_filepath(document)    
    
    # Prepare text answer
    answer_text = f"""{res}

Source: {doc}
Page: {page}"""

    # Convert PIL `image` to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return a dict instead of a single string
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
    # Return JSON to the client
    return jsonify(response_data)

if __name__ == "__main__":
    app.run(port=5000)
