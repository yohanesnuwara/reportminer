![image](https://github.com/user-attachments/assets/c9bcf0b6-5335-49ea-9b9a-eba9d51feb40)

ReportMiner is an easy-to-use wrapper for multimodal retrieval augmented generation (RAG) tasks for technical reports. With ReportMiner, you can use any HuggingFace or Byaldi and apply for your reports. 

## 🤖 Installation
Make sure to install Poppler first

`sudo apt-get install -y poppler-utils` on Linux OS

Then, install the package

`pip install reportminer`

## 🚀 Use RAG in 3 simple steps

Using ReportMiner is just few lines of codes. 

### 1. Setup embedding model and visual language model

Here we will use ColPali-1.2 as our embedding model and SmolVLM as our visual language model (by default). This combination works very well even with free-tier Colab GPU T4 and low memory.

```
from reportminer import rag

# Setup embedding and VL model
rag_models = rag.setup_model()
```

If you have access to Colab Pro's GPU A100 and high memory, you can make advantage of Qwen2-VL as your visual language model and activate Flash Attention 2.

### 2. Upload your report

```
# Specify pdf file
pdf_file = '/content/15-9-19a-core.pdf'

# Process PDF report by converting into embeddings
rag_models = rag.process_PDF(pdf_file, rag_models, dpi=200, index_name='pvt-rag')
```

### 3. Perform query

There are 2 minimal tasks you can use with ReportMiner. First, you can retrieve relevant page taking advantage of the embedding model

```
rag.RAG('How is the Klinkenberg corrected gas horizontal permeability to porosity looks like at core 7?', 
        rag_models, k=1)
```

Second, you can ask a question, taking advantage of the visual language model

```
rag.Ask('How many cored intervals are there in the report? Mention the depth of each', rag_models)
```

## 👨‍💻 Future development

In the next version, the following will be implemented:

1. Support for multiple PDF reports in a directory structure (best industrial case)
2. Support for fine tuning with user reports of specific domain
3. Integration with vector databases for robust document indexing 



