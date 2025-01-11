# ReportMiner
ReportMiner is an easy-to-use wrapper for multimodal retrieval augmented generation (RAG) tasks for technical reports. With ReportMiner, you can use any HuggingFace or Byaldi and apply for your reports. 

## Installation
Make sure to install Poppler first

`sudo apt-get install -y poppler-utils` on Linux OS

Then, install the package

`pip install reportminer`

## Use RAG in 3 simple steps

Here we will use ColPali-1.2 as our embedding model and SmolVLM as our visual language model. This combination works very well even with free-tier Colab GPU T4 and low memory.

If you have access to Colab Pro's GPU A100 and high memory, you can make advantage of Qwen2-VL as your visual language model and activate Flash Attention 2.

### 1. Setup embedding model and visual language model

### 2. Upload your report

### 3. Perform query

## Future development

In the next version, the following will be implemented:

✅ Support for multiple PDF reports in a directory structure (best industrial case)

✅ Support for fine tuning with user reports of specific domain

✅ Integration with vector databases for robust document indexing 



