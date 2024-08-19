**Medical Project Documentation**

**Title**: AI-Based Clinical Data Management System
**Name**: Polavarapu Bhavita 
**Internship Position**: Generative AI Intern
**Date of Submission:** August 19, 2024

**Acknowledgements**

I would like to express my gratitude to my mentor Mr. Chakrahari Aravind Sai, for their invaluable guidance and support throughout this project. I also thank my colleagues for their collaboration and encouragement.


**Introduction**

Access to well-organized clinical data is essential for effective healthcare delivery. This project aims to enhance the usability of clinical data through advanced AI techniques, focusing on entity extraction, report summarization, and a chatbot interface for user interaction. The system is designed to assist healthcare professionals in quickly accessing and understanding patient information, thereby improving decision-making and patient care.


**Project Objectives**

The main objectives of this project include:
Entity Extraction: To identify and extract key medical entities from clinical documents, such as patient names, ages, diseases, treatments, and symptoms.
Report Summarization: To summarize lengthy clinical reports into concise and relevant insights for quick understanding.
Chatbot Development: To create a chatbot that can answer questions based on clinical documents and general medical inquiries, enhancing user accessibility and information retrieval.


**Project Files and Usage**

venv: This virtual environment is used to manage dependencies, ensuring that the project runs with specific library versions without affecting the global Python environment.
config.yml: This configuration file specifies the models and tokenizers used in the project.
utils.py: This module contains utility functions and classes that support the main functionalities of the application, including:
EntityExtractor: Extracts medical entities from clinical text, with methods for reading PDF files, preprocessing text, and extracting patient details.
Summarizer: Generates summaries of clinical reports using the T5 model, including methods for preprocessing input files and creating focused prompts for summarization.
answer_pdf_question: Provides answers to user queries based on the content of uploaded PDF documents using a question-answering model.
get_response: Handles general inquiries using the LLaMA2 model.
generate_audio: Converts text to speech using the gTTS library and saves the output as an MP3 file.
play_audio: Plays the generated audio file using system commands.
extract_text_from_image: (OCR functionality) Uses an OCR library Tesseract to convert images of text into machine-readable text, allowing the application to process scanned documents.
app.py: The main application file that sets up the Flask web server and handles user interactions, including:
Routes: Defines endpoints for entity extraction, summarization, and chatbot, ocr interactions.
File Handling: Manages the uploading of PDF files and processes them using methods defined in utils.py.
index.html: 
The index.html file serves as the front-end interface for the application, providing a user-friendly layout for interacting with the AI-based clinical data management system.
Document Upload Section: A file input box for users to upload clinical documents in PDF format, with clear instructions on accepted file types.
Loader Animation: A visual loader that appears during document uploads or query submissions, indicating that the application is processing the request.
Input box: The interface features an input box for users to ask questions about clinical data, with chatbot responses displayed dynamically. It also shows extracted information from uploaded documents, including patient details and a summary of the clinical report, all on the same page of the web interface.
Audio Playback Controls: Controls for playing audio generated from text-to-speech, enabling users to listen to extracted entities, chatbot replies and summaries.
Styling and Responsiveness: The interface is styled with CSS for a clean appearance.


**Libraries and Frameworks**

Flask: A lightweight web framework used to create the web application, manage routes, and handle user interactions.
PyPDF2: A library for reading and extracting text from PDF files, enabling the application to process clinical documents uploaded by users.
gTTS (Google Text-to-Speech): A library that converts text into spoken audio, allowing the application to provide audio feedback for extracted entities and summaries.
Torch: A deep learning framework used to load and run the pre-trained models for Named Entity Recognition (NER) and question answering.
Transformers (Hugging Face): A library that provides pre-trained models and tokenizers for various natural language processing tasks, including:
AutoTokenizer: Used to tokenize input text for the NER model.
AutoModelForTokenClassification: Utilized for the NER task to classify tokens in the text.
T5Tokenizer and T5ForConditionalGeneration: Employed for generating summaries of clinical reports.
DistilBertTokenizer and DistilBertForQuestionAnswering: Used for handling question-answering tasks.
MBZUAI/LaMini-Flan-T5-248M: The model used for summarization, designed for generating concise summaries from clinical reports.
LLaMA2 Model: Used for general inquiry responses, allowing the chatbot to handle a wide range of questions.
Langchain: A library that simplifies the handling of text documents and supports splitting text into manageable chunks for processing. It includes:
RecursiveCharacterTextSplitter: Used to split long text documents into smaller sections for easier summarization.
PyPDFLoader: A document loader that facilitates reading and splitting PDF files.
Subprocess: A built-in Python module used to run external commands, enabling the application to interact with other models or processes, such as invoking the LLaMA2 model for general inquiries.
YAML: A data serialization format used to read configuration settings from the config.yml file, allowing for easy management of model parameters and settings.
Tesseract (OCR Library): An OCR library that can be integrated to convert images of text into machine-readable text, enabling the application to process scanned documents effectively.


**Project Components**

.1Entity Extraction
The entity extraction component is designed to identify and extract significant medical entities from clinical text. This involves using a Named Entity Recognition (NER) system that leverages pre-trained models to recognize specific terms related to patient information.
Model used: 
The model utilized for entity extraction is SciBERT, a variant of BERT (Bidirectional Encoder Representations from Transformers) specifically pre-trained on scientific text. SciBERT is particularly effective for biomedical and clinical applications due to its training on a large corpus of scientific literature. The model is initialized in the EntityExtractor class using the configuration from the config.yml file:
Integration Points:
The EntityExtractor class is initialized with a configuration file that specifies the model and tokenizer to be used. This class is responsible for loading the necessary models and defining the entities of interest (diseases, treatments, and symptoms).
The method extract_text_from_pdf utilizes the PyPDF2 library to read and extract text from uploaded PDF files, ensuring that the data is in a usable format for further processing.
Code Modules and Functions:
extract_text_from_pdf(pdf_path): Reads a PDF file and extracts text from each page.
preprocess_text(text): Cleans the extracted text by removing unnecessary whitespace.
extract_entities(text): Processes the text to identify and classify entities using the loaded NER model. The method returns a structured dictionary containing the patient's name, age, diseases, treatments, and symptoms.
process_entities(text, entities): Uses regular expressions to extract specific details such as patient names and ages from the text.
2 Report Summary
The report summarization component aims to condense lengthy clinical reports into concise summaries that highlight key patient information.
 Model Used:
 The summarization component employs the MBZUAI/LaMini-Flan-T5-248M model, which is designed for text generation tasks, particularly effective in summarizing clinical reports.
Integration Points:
The Summarizer class is initialized with the same configuration file as the EntityExtractor, allowing it to utilize the entity extraction capabilities during the summarization process.
The method file_preprocessing uses the Langchain library to load and split PDF documents into manageable chunks for summarization.
Code Modules and Functions:
file_preprocessing(file): Loads the PDF file and splits it into smaller sections for easier processing.
summarize(filepath): Extracts patient information using the EntityExtractor and creates a focused prompt for the summarization model. It generates a summary that includes key patient details and condenses the clinical report into a coherent paragraph.
This summarization capability is particularly beneficial for healthcare professionals who need quick access to critical patient information without sifting through extensive documentation.
.3 Chatbot Implementation
The chatbot serves as an interactive interface that allows users to ask questions related to clinical data and receive informative responses.
Model Used:  
The chatbot utilizes two distinct models:
 DistilBERT: Used for answering questions based on the content of uploaded PDF documents through the answer_pdf_question function.LLaMA2: Employed for general inquiries, allowing users to ask questions that are not necessarily tied to the uploaded documents via the get_response function.
Integration Points:
The chatbot is implemented as a Flask route that handles user input and integrates with both the entity extraction and summarization components.
It employs two distinct functionalities: PDF question-answering and general question-answering, utilizing different models for each purpose.
Code Modules and Functions:
answer_pdf_question(question, pdf_text): Utilizes the DistilBERT model to answer questions based on the content of the uploaded PDF document. It extracts relevant context and generates answers.
get_response(model_name, user_input): Uses the LLaMA2 model to handle general inquiries, allowing users to ask questions that are not necessarily tied to the uploaded documents.
is_medical_question(question): A helper function that determines if a question is related to medical content by checking for specific keywords.
The chatbot enhances user engagement by providing immediate responses to inquiries, thereby facilitating better communication and understanding of clinical data.


**Additional Features**

.1 OCR Integration
To accommodate scanned documents that may not be in a text format, the system integrates Optical Character Recognition (OCR) capabilities using the Tesseract library.
Model Used:
 The system employs Tesseract for OCR, enabling the conversion of images of text into machine-readable text.
Integration Points:
The OCR functionality is invoked before the entity extraction and summarization processes, ensuring that all document types are usable.
Code Modules and Functions:
extract_text_from_image(image_path): This function utilizes the Tesseract OCR library to convert images of text into machine-readable text, taking an image file path as input and returning the extracted text.
This integration ensures that the system can process both PDF documents and scanned images, providing a comprehensive solution for handling various types of clinical data.
.2 Text-to-Speech Functionality
The system includes a text-to-speech feature that allows users to listen to the extracted entities and summaries.
Model Used:
The system employs the gTTS (Google Text-to-Speech) library to convert text into audio files.
Code Modules and Functions:
generate_audio(text, language): Uses the gTTS library to convert text into an audio file, which can then be played back to the user.
play_audio(file_path): Plays the generated audio file using the appropriate system command based on the operating system.
This functionality is particularly useful for users who prefer auditory information or for accessibility purposes.


**Challenges Faced**

Data Quality Issues: Ensuring that the input PDFs were of high quality for text extraction was a challenge. Some documents had poor formatting or were scanned at low resolutions, which affected the OCR performance.
Integration Complexity: Coordinating multiple components (NER, summarization, chatbot) into a cohesive application required careful planning and testing to ensure that all parts worked together smoothly.


**Results**

Entity Extraction:
The EntityExtractor class successfully identifies and extracts key medical entities from clinical documents. The entities extracted include patient names, ages, diseases, treatments, and symptoms. 
The method extract_entities uses a combination of regular expressions and a pre-trained model to identify entities. 
Report Summarization:
The Summarizer class generates concise summaries of clinical reports by utilizing the T5 model. The summaries include essential patient details, medical history, current medications, symptoms, and treatment plans.
Chatbot Performance:
The chatbot functionality, implemented through the Flask application, allows users to ask questions related to clinical data. The chatbot uses the answer_pdf_question function to provide answers based on the content of uploaded PDF documents and the get_response function for general inquiries.


**Conclusions**

The project successfully demonstrated the potential of AI in enhancing the usability of clinical data. The integrated system allows for efficient entity extraction, summarization, and user interaction through a chatbot. And it also allowed the user the freedom to use any type of document format. This tool can significantly assist healthcare professionals in managing and understanding patient information, ultimately improving patient care.


**Future Work**

Expand NER capabilities to include additional medical entities and relationships, such as family medical history, allergies, and lab results, enhancing the understanding of patient data.
Improve the chatbot's conversational abilities by implementing advanced natural language processing techniques, enabling it to manage complex queries and provide contextually relevant responses.
Add multilingual support to cater to a broader audience, improving accessibility for non-English speaking users and enhancing user engagement.
Integrate lip-sync capabilities for the text-to-speech feature to create a more engaging user interface, making interactions with the chatbot more immersive.
Optimize response time by enhancing backend processing, reducing latency in data retrieval and response generation for quicker user feedback.
Fine-tune models for problem-oriented results by adjusting them based on specific use cases, ensuring outputs align better with user needs and clinical scenarios.


**References**
https://huggingface.co/

https://ollama.com/

https://github.com/dsdanielpark/gpt2-bert-medical-qa-chat/







