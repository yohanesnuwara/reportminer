import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for generating PNG image
import io
import base64
import numpy as np

from reportminer import rag, rag_folder

# Load colpali model
print('Loading pretrained embedding ...')
model = rag.load_model_from_pretrained('/home/reportminer/.byaldi/rag')

def chatbot_response(msg):
    # Ask the message
    res, source, image = rag_folder.Ask(msg, model)
    print('SOURCE:', source)
    plt.show(np.array(image))
    return res


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run(port=5001)