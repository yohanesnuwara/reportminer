import argparse
import os
import datetime
import io
import base64
import csv

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, flash
)
import matplotlib
matplotlib.use('Agg')
from reportminer import rag, rag_folder

app = Flask(__name__)
app.static_folder = 'static'

# ─── SECRET KEY & MAGIC FILE ────────────────────────────────────────────────
app.secret_key = "replace_this_with_a_strong_secret"

# Load credentials from magic.txt
CREDENTIALS = {}
with open("magic.txt") as mf:
    reader = csv.DictReader(mf)
    for row in reader:
        CREDENTIALS[row["USERNAME"]] = row["PASSWORD"]


# ─── RAG SETUP ───────────────────────────────────────────────────────────────
rag_models = None

def run_embedding(base_dir):
    global rag_models
    print('Setting up RAG models...')
    rag_models = rag.setup_model2()

    print('Normalizing folder structure...')
    rag_folder.normalize_folder_structure(base_dir, base_dir)

    print('Embedding documents...')
    start = datetime.datetime.now()
    rag_models = rag_folder.Process(base_dir, rag_models)
    print('Embedding completed in:', datetime.datetime.now() - start)


def chatbot_response(msg):
    responses, sources, images = rag_folder.Ask_iterative(msg, rag_models, k=2)
    answer_parts, image_list = [], []

    for i, res in enumerate(responses):
        doc_path, page, _ = sources[i]
        orig_file = rag_folder.retrieve_original_filepath(doc_path)
        text_seg = f"""{res}

**Source**: {orig_file}
**Page**: {page}"""
        answer_parts.append(text_seg)

        buf = io.BytesIO()
        images[i].save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_list.append(img_b64)

    return {
        "answer": "\n\n".join(answer_parts),
        "images": image_list
    }


# ─── LOGIN ROUTES ───────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        pwd  = request.form["password"]
        if CREDENTIALS.get(user) == pwd:
            session["user"] = user
            return redirect(url_for("chat_ui"))
        flash("Invalid username or password", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─── CHAT UI & API ──────────────────────────────────────────────────────────

@app.route("/app")
def chat_ui():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    if "user" not in session:
        return jsonify({"error": "not authenticated"}), 401
    userText = request.args.get("msg", "")
    return jsonify(chatbot_response(userText))


# ─── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask chatbot with embedding.")
    parser.add_argument('-i', required=True, help='Folder path to embed')
    args = parser.parse_args()

    # 1) Do embedding BEFORE starting Flask
    run_embedding(args.i)

    # 2) Start Flask WITHOUT the auto‑reloader so embedding only ran once
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
