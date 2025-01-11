from setuptools import setup

setup(
    name='ReportMiner',
    version='0.1.0',    
    description='Easy to use wrapper for text document question and answer using RAG and LLM',
    url='https://github.com/yohanesnuwara/reportminer',
    author='Yohanes Nuwara',
    author_email='ign.nuwara97@gmail.com',
    license='MIT License',
    packages=['reportminer'],
    install_requires=[
        'byaldi>=0.0.7',
        'transformers>=4.46.3',
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
