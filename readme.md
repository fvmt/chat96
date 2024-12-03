# Chat96

## Introduction
------------
This app is an attempt to solve QA ML challenge
## Installation
------------

You need Ollama installed in the system with the following LLMs:
llama3.2
Download the following Embeddings from HuggingFace:
multi-qa-MiniLM-L6-cos-v1
```
   pip install -r requirements.txt
   ```
Set the following env variable:
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
This is required because of streamlit conflicting dependencies
and streamlit execution doesn't support dotenv out of the box

Also you'll need Qdrant db

Enter valid settings into .env file:
```
OPENAI_API_KEY=

HUGGINGFACEHUB_API_TOKEN=
COLLECTION_NAME=Aparavi96

UNSTRUCTURED_API_KEY=
UNSTRUCTURED_API_URL=

EMBEDDING_MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1

QDRANT_URL=http://127.0.0.1:16333
```
Please note that this was written with python 3.12

## Execution
-----
To use the app, follow these steps:

1. Ensure that you have installed the required dependencies
and added all the settings to the `.env` file.

2. Run the `app.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Enter the directory with documents into the corresponding textbox and click process. This will set up the vector collection and prepare data using unstructured.io. Some of the files are supported by default, so try working with .txt and .pdf files
5. If you'll ask a question before clicking 'Process' the app will crash for now. 
6. Also, there is a console mode client, which doesn't require processing.

