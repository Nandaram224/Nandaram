
# Steps AI NLP Intern (2) Application Update

Objective:

The objective of this assessment is to assess your ability to extract content from selected textbooks, create a hierarchical tree-based index for multi-document/topic/section-based RAG, and develop a question-answering system using an LLM (Language Model). The assessment will evaluate your skills in data extraction, data processing, hierarchical tree-based indexing, retrieval techniques, and natural language processing.

Usage:

Text Extraction:

import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)): # Use len(reader.pages) to get number of pages
            page = reader.pages[page_num] # Access pages using index
            text += page.extract_text()
    return text

pdf_path = '/content/Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf'
text = extract_text_from_pdf(pdf_path)
print(text[:100])  # Print the first 2000 characters to verify extraction

Extract Text to File:

import fitz  # fitz is the PyMuPDF library

def extract_text_to_file(pdf_path, output_txt_path):
    doc = fitz.open(pdf_path)
    with open(output_txt_path, 'w', encoding='utf-8') as output_file:
        for page in doc:
            text = page.get_text()
            output_file.write(text)

# Example usage
pdf_path = "/content/Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf"  # Replace with the actual path
output_txt_path = "/content/extracted_text.txt"  # Choose the output file path
extract_text_to_file(pdf_path, output_txt_path)
print("Text extracted and saved to:", output_txt_path)

Hierarchical Indexing:

import re
from nltk.tokenize import sent_tokenize

class HierarchicalIndexer:
    def __init__(self, text):
        self.text = text
        self.index = []

    def create_index(self):
        chapters = re.split(r'CHAPTER \d+', self.text)
        for chapter in chapters:
            if chapter.strip():
                sections = re.split(r'Section \d+\.\d+', chapter)
                chapter_index = {'sections': []}
                for section in sections:
                    if section.strip():
                        subsections = re.split(r'Subsection \d+\.\d+\.\d+', section)
                        section_index = {'subsections': []}
                        for subsection in subsections:
                            if subsection.strip():
                                sentences = sent_tokenize(subsection)
                                subsection_index = {'content': sentences}
                                section_index['subsections'].append(subsection_index)
                        chapter_index['sections'].append(section_index)
                self.index.append(chapter_index)
        return self.index

indexer = HierarchicalIndexer(text)
hierarchical_index = indexer.create_index()
for chapter in hierarchical_index[:5]:  # Print the first 5 chapters to verify indexing
    print("Chapter:")
    for section in chapter['sections']:
        print("  Section:")
        for subsection in section['subsections']:
            print("    Subsection content:")
            for sentence in subsection['content']:
                print(f"      {sentence}")


Content Retrieval:

from sentence_transformers import SentenceTransformer, util
import torch

# Dense Passage Retrieval with Sentence-Transformers
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

def dense_retrieve(query, passages):
    query_embedding = model.encode(query, convert_to_tensor=True)
    passage_embeddings = model.encode(passages, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, passage_embeddings)
    results = sorted(enumerate(scores[0]), key=lambda x: x[1], reverse=True)
    return [(passages[idx], score.item()) for idx, score in results]

def retrieve_content(query, hierarchical_index):
    passages = []
    for chapter in hierarchical_index:
        for section in chapter['sections']:
            for subsection in section['subsections']:
                passages.extend(subsection['content'])
    dense_results = dense_retrieve(query, passages)
    return sorted(dense_results, key=lambda x: x[1], reverse=True)[:10]

query = "What is supervised learning?"

retrieved_content = retrieve_content(query, hierarchical_index)
print("Content retrieved successfully.")
for content, score in retrieved_content:
    print(f"{content} (Score: {score})")


Question Answering:

from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def generate_answer(query, retrieved_content):
    context = "\n".join([content for content, score in retrieved_content])
    print(f"Context:\n{context}\n")  # Print the context to verify it contains enough information
    answer = qa_pipeline(question=query, context=context)
    return answer['answer']

query = "What is supervised learning?"
retrieved_content = retrieve_content(query, hierarchical_index)
answer = generate_answer(query, retrieved_content)
print(f"Answer:\n{answer}")


Streamlit App:

import PyPDF2
import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import streamlit as st

# Step 1: Text Extraction
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

pdf_path = '/content/Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf'
text = extract_text_from_pdf(pdf_path)
if text:
    st.write("Text extracted successfully.")
else:
    st.write("Failed to extract text.")
    st.stop()

# Step 2: Hierarchical Tree-Based Indexing
class HierarchicalIndexer:
    def __init__(self, text):
        self.text = text
        self.index = []

    def create_index(self):
        chapters = re.split(r'CHAPTER \d+', self.text)
        for chapter in chapters:
            if chapter.strip():
                sections = re.split(r'Section \d+\.\d+', chapter)
                chapter_index = {'sections': []}
                for section in sections:
                    if section.strip():
                        subsections = re.split(r'Subsection \d+\.\d+\.\d+', section)
                        section_index = {'subsections': []}
                        for subsection in subsections:
                            if subsection.strip():
                                sentences = sent_tokenize(subsection)
                                subsection_index = {'content': sentences}
                                section_index['subsections'].append(subsection_index)
                        chapter_index['sections'].append(section_index)
                self.index.append(chapter_index)
        return self.index

indexer = HierarchicalIndexer(text)
hierarchical_index = indexer.create_index()
if hierarchical_index:
    st.write("Index created successfully.")
    st.write(f"Number of chapters indexed: {len(hierarchical_index)}")
    st.write(f"Index details (first 2 chapters):")
    for chapter in hierarchical_index[:2]:
        st.write("Chapter:")
        for section in chapter['sections']:
            st.write("  Section:")
            for subsection in section['subsections']:
                st.write("    Subsection content:")
                for sentence in subsection['content']:
                    st.write(f"      {sentence}")
else:
    st.write("Failed to create index.")
    st.stop()

# Step 3: Content Retrieval
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

def dense_retrieve(query, passages):
    query_embedding = model.encode(query, convert_to_tensor=True)
    passage_embeddings = model.encode(passages, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, passage_embeddings)
    results = sorted(enumerate(scores[0]), key=lambda x: x[1], reverse=True)
    return [(passages[idx], score.item()) for idx, score in results]

def retrieve_content(query, hierarchical_index):
    passages = []
    for chapter in hierarchical_index:
        for section in chapter['sections']:
            for subsection in section['subsections']:
                passages.extend(subsection['content'])
    st.write(f"Total passages to retrieve from: {len(passages)}")
    dense_results = dense_retrieve(query, passages)
    return sorted(dense_results, key=lambda x: x[1], reverse=True)[:10]

query = st.text_input("Enter your query:")
if query:
    st.write("Query received.")
    retrieved_content = retrieve_content(query, hierarchical_index)
    if retrieved_content:
        st.write("Content retrieved successfully.")
        st.write("Retrieved content:")
        for content, score in retrieved_content:
            st.write(f"Content: {content}")
            st.write(f"Score: {score}")

        # Step 4: Question Answering
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

        def generate_answer(query, retrieved_content):
            context = "\n".join([content for content, score in retrieved_content])
            st.write(f"Context provided to the model:\n{context}\n")
            answer = qa_pipeline(question=query, context=context)
            return answer['answer']

        answer = generate_answer(query, retrieved_content)
        st.write(f"Answer:\n{answer}")
    else:
        st.write("No content retrieved.")
else:
    st.write("No query provided.")

# Display query for debugging
st.write(f"Query: {query}")


Deploy with ngrok:
import os
import subprocess
from pyngrok import ngrok

# Remove any previously set authtokens (just in case)
!ngrok config remove-authtoken

# Set your authtoken (replace with your actual token)
NGROK_AUTHTOKEN = "2it2qeI5LXvbvqa3k9RhmHADUx4_XApT25rcLcsMt8BTtbff"  # Make sure this is the correct token
os.environ["NGROK_AUTHTOKEN"] = NGROK_AUTHTOKEN

# Verify that the authtoken is set correctly
!ngrok authtoken $NGROK_AUTHTOKEN  # This should print a success message

# Start Streamlit server and capture its output
streamlit_proc = subprocess.Popen(['streamlit', 'run', '/content/textbook_qa.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Connect to ngrok and get the public URL, explicitly specifying the port
public_url = ngrok.connect(8501) # Pass the port as a number, not a string
print(f"Streamlit server started successfully. Access your app at {public_url}")

# Print Streamlit logs
print("Streamlit logs:")
while True:
    streamlit_output = streamlit_proc.stdout.readline().strip()
    if streamlit_output:
        print(streamlit_output)
    if streamlit_proc.poll() is not None:
        break


The textbook of "Introduction to Machine Learning with Python" by Author Andreas C. Müller & Sarah Guido is used in the project.

## 🚀 About Me
An Aspiring Data Scientist and machine learning enthusiast with a keen interest in natural language processing and AI applications. My background includes extensive experience in Python programming, data analysis, and developing machine learning models. I created this project to explore the intersection of text extraction, hierarchical indexing, and question-answering systems.

Through this project, I aim to provide an efficient and user-friendly way to extract, index, and retrieve relevant information from PDF documents using advanced NLP techniques. I believe in the power of open-source collaboration and am excited to see how this project can evolve with contributions from the community.

Connect with Me:
GitHub: https://github.com/Nandaram224/Nandaram
LinkedIn: https://www.linkedin.com/in/nanda-ram-aba320128/
Email: Nandaramlsj@gmail.com

