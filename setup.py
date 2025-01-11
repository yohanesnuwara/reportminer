from setuptools import setup
import shutil

# Check if `poppler` is installed
if not shutil.which("pdftotext"):  # `pdftotext` is a Poppler utility
    print(
        "Warning: Poppler is not installed. "
        "Please install it using your system's package manager:\n"
        "  On Ubuntu/Debian: sudo apt-get install -y poppler-utils\n"
        "  On macOS: brew install poppler\n"
    )

with open("reportminer/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ReportMiner',
    version='0.1.1',    
    description='Easy to use wrapper for text document question and answer using RAG and LLM',
    long_description=long_description,  
    long_description_content_type="text/markdown",  
    url='https://github.com/yohanesnuwara/reportminer',
    author='Yohanes Nuwara',
    author_email='ign.nuwara97@gmail.com',
    license='MIT License',
    packages=['reportminer'],
    install_requires=[
        'byaldi>=0.0.7',
        'transformers>=4.46.3',
        'pdf2image>=1.17.0',
        'pillow>=11.1.0',
        'numpy>=1.26.4',
        'matplotlib>=3.10.0'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10'
    ],
)
