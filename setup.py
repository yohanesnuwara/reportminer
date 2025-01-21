from setuptools import setup
import shutil
import os

# Check if `poppler` is installed
if not shutil.which("pdftotext"):  # `pdftotext` is a Poppler utility
    print(
        "Warning: Poppler is not installed. "
        "Please install it using your system's package manager:\n"
        "  On Ubuntu/Debian: sudo apt-get install -y poppler-utils\n"
        "  On macOS: brew install poppler\n"
    )

# Get the directory of this script
current_dir = os.path.abspath(os.path.dirname(__file__))

# Path to README.md
readme_path = os.path.join(current_dir, "README.md")

# Read the README.md content
with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ReportMiner',
    version='0.2.2',
    description='Easy to use wrapper for text document question and answer using RAG and LLM',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/yohanesnuwara/reportminer',
    author='Yohanes Nuwara',
    author_email='ign.nuwara97@gmail.com',
    license='MIT License',
    packages=['reportminer'],

    # Minimal install for basic functionality
    install_requires=[
        'byaldi>=0.0.7',
        'transformers>=4.46.3',
        'pdf2image>=1.17.0',
        'pillow>=11.1.0',
        'numpy>=1.26.4',
        'matplotlib>=3.10.0',
        'pypdf2>=3.0.1'
    ],

    # Optional extras
    extras_require={
        # Install with `pip install reportminer[ppt]`
        'ppt': [
            'aspose.slides>=24.12.0',
            'spire.presentation>=9.12.1'
        ],

        # Install with `pip install reportminer[xls]`
        'xls': [
            'aspose-cells-python>=25.1.0',
        ],

        # Install both PPT and XLS features with `pip install reportminer[all]`
        # (optional "all" extra if you want everything at once)
        'all': [
            'aspose.slides>=24.12.0',
            'spire.presentation>=9.12.1',
            'aspose-cells-python>=25.1.0'
        ],
    },

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10'
    ],
)
