import yaml
from gtts import gTTS
import os
import re
import torch
import os 
import PyPDF2
import subprocess
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

class EntityExtractor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer_name'], max_length=512, truncation=True)
        self.model = AutoModelForTokenClassification.from_pretrained(config['model']['model_name'])
        self.ner_pipeline = pipeline('ner', model=self.model, tokenizer=self.tokenizer)

        self.diseases = [
            'diabetes', 'hypertension', 'cancer', 'asthma', 'dementia', 
            'stroke', 'heart disease', 'kidney disease', 'liver disease'
        ]
        self.treatments = [
            'insulin', 'chemotherapy', 'radiation therapy', 'physical therapy', 
            'surgery', 'medication', 'therapy', 'counseling', 'rehabilitation', 
            'donepezil', 'lisinopril', 'aspirin'
        ]
        self.symptoms = [
            'memory loss', 'confusion', 'difficulty with daily tasks', 'weakness', 
            'difficulty walking', 'high blood pressure'
        ]

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        return text.strip()

    def preprocess_text(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def extract_entities(self, text):
        text = self.preprocess_text(text)
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        outputs = self.model(**tokens)
        predictions = torch.argmax(outputs.logits, dim=2)

        entities = []
        for token, prediction in zip(tokens['input_ids'][0], predictions[0]):
            if prediction.item() != 0:  # Assuming 0 is the index for 'O' (no entity)
                entities.append(self.tokenizer.decode([token.item()]))

        patient_name, patient_age, patient_diseases, patient_treatments, patient_symptoms = self.process_entities(text, entities)
        
        return {
            'name': patient_name,
            'age': patient_age,
            'diseases': list(patient_diseases),
            'treatments': list(patient_treatments),
            'symptoms': list(patient_symptoms)
        }

    def process_entities(self, text, entities):
        patient_name = None
        patient_age = None
        patient_diseases = set()
        patient_treatments = set()
        patient_symptoms = set()

        name_pattern = re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b')
        name_matches = name_pattern.findall(text)
        
        if name_matches:
            unique_names = set(name_matches)
            patient_name = next(iter(unique_names), None)

        for disease in self.diseases:
            if disease.lower() in text.lower():
                patient_diseases.add(disease)

        for treatment in self.treatments:
            if treatment.lower() in text.lower():
                patient_treatments.add(treatment)

        for symptom in self.symptoms:
            if symptom.lower() in text.lower():
                patient_symptoms.add(symptom)

        age_pattern = re.compile(
            r'\b(?:age|aged|years?|yrs?)\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:years?|yrs?)\b', re.IGNORECASE)
        age_matches = age_pattern.findall(text)
        
        if age_matches:
            patient_age = next((match[0] or match[1] for match in age_matches if match), None)

        return patient_name, patient_age, patient_diseases, patient_treatments, patient_symptoms

distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

def preprocess_text(pdf_text):
    sections = pdf_text.split('section')
    relevant_text = ""
    for section in sections:
        if 'patient' in section.lower() or 'doctor' in section.lower():
            relevant_text += section.strip() + " "
    return relevant_text

def chunk_text(text, max_length):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])


def answer_pdf_question(question, pdf_text):
   
    model_id = "distilbert-base-cased-distilled-squad"
    pipe = pipeline("question-answering", model=model_id)

    res = pipe(question=question, context=pdf_text)
    return res["answer"]


def get_response(model_name, user_input):

    input_text = user_input
    
    try:
        result = subprocess.run(
            ['ollama', 'run', model_name, input_text],
            capture_output=True,  
            text=True,
            encoding='utf-8',
            check=True  
        )

        output = result.stdout.strip()
        
        output_lines = output.split('\n')
        limited_output = '\n'.join(output_lines[:4]) 
        
        if not limited_output.strip():
            return "I'm sorry, I don't have a response for that query."
        
        return limited_output.strip() 
    
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")  
        return "An error occurred while processing your request. Please try again."
    
    except Exception as e:
        print(f"Exception: {str(e)}") 
        return "An error occurred while processing your request. Please try again."

def is_medical_question(question):
    """Determine if the question is related to medical content."""
    medical_keywords = ['symptom', 'diagnosis', 'treatment', 'medication', 'report', 'test', 'results', 'condition']
    return any(keyword in question.lower() for keyword in medical_keywords)

class Summarizer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.tokenizer = T5Tokenizer.from_pretrained(config['model']['summarizer_tokenizer_name'])
        self.model = T5ForConditionalGeneration.from_pretrained(config['model']['summarizer_model_name'])
        self.entity_extractor = EntityExtractor(config_path)  # Initialize EntityExtractor

    def file_preprocessing(self, file):
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = text_splitter.split_documents(pages)
        final_texts = ""
        for text in texts:
            final_texts += text.page_content
        return final_texts

    def summarize(self, filepath):
        input_text = self.file_preprocessing(filepath)

        entities = self.entity_extractor.extract_entities(input_text)

        prompt = (
            f"Patient Report:\n\n"
            f"Name: {entities['name']}\n"
            f"Age: {entities['age']}\n"
            f"Medical Conditions: {', '.join(entities['diseases'])}\n"
            f"Medications: {', '.join(entities['treatments'])}\n"
            f"Symptoms: {', '.join(entities['symptoms'])}\n\n"
            "Summarize the detailed information above into a single coherent paragraph. "
            "Include the patient's personal details, medical history, current medications, symptoms, and treatment plans. "
            "Make sure the summary is concise and to the point.\n\n"
            f"{input_text}"
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
        
        output = self.model.generate(input_ids, max_length=800, num_return_sequences=1, num_beams=4, early_stopping=True)
        
        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return summary
    

def generate_audio(text, language):
    """
    Generate audio from text using gTTS and save it as an MP3 file.
    """
    tts = gTTS(text=text, lang=language, slow=False)
    audio_file = "output.mp3"
    tts.save(audio_file)

    return audio_file

def play_audio(file_path):
    """
    Play the generated audio file.
    """
    if os.name == 'nt':  
        os.startfile(file_path)
    else:
        opener = "open" if os.name == "posix" else "xdg-open"
        os.system(f"{opener} {file_path}")