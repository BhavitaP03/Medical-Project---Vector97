from flask import Flask, request, render_template, jsonify
from utils import EntityExtractor, Summarizer, answer_pdf_question, get_response, is_medical_question, generate_audio, play_audio
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract_entities', methods=['POST'])
def extract_entities():
    pdf_file = request.files['pdf_file']
    if pdf_file:
        uploads_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        pdf_path = os.path.join(uploads_dir, pdf_file.filename)
        pdf_file.save(pdf_path)
        
        extractor = EntityExtractor('config.yml')
        try:
            text = extractor.extract_text_from_pdf(pdf_path)
            entities = extractor.extract_entities(text)
            
            entity_info = (
                f"Patient Name: {entities['name']}\n"
                f"Patient Age: {entities['age']}\n"
                f"Patient Diseases: {', '.join(entities['diseases'])}\n"
                f"Patient Treatments: {', '.join(entities['treatments'])}"
            )
            
            return entity_info
        except Exception as e:
            return f"Error extracting entities: {e}"
    else:
        return "No file uploaded."

@app.route('/summarize', methods=['POST'])
def summarize():
    pdf_file = request.files['pdf_file']
    if pdf_file:
        uploads_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        pdf_path = os.path.join(uploads_dir, pdf_file.filename)
        pdf_file.save(pdf_path)
        
        summarizer = Summarizer('config.yml')
        try:
            summary = summarizer.summarize(pdf_path)
            return summary
        except Exception as e:
            return f"Error generating summary: {e}"
    else:
        return "No file uploaded."


@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.form['user_input']
    pdf_file = request.files.get('pdf_file')  
    pdf_text = ""
    
    if pdf_file:
        uploads_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        pdf_path = os.path.join(uploads_dir, pdf_file.filename)
        pdf_file.save(pdf_path)
        
        extractor = EntityExtractor('config.yml')
        try:
            pdf_text = extractor.extract_text_from_pdf(pdf_path)
            print(f"Extracted PDF Text: {pdf_text}")  
        except Exception as e:
            return jsonify({'error': f"Error extracting text from PDF: {e}"})
    
    print(f"User Input: {user_input}")   
    print(f"PDF Text: {pdf_text}") 

    if not pdf_text:
        return jsonify({'error': 'No context available from PDF for answering the question.'})

    response = answer_pdf_question(user_input, pdf_text)
    
    return jsonify({'response': response})

@app.route('/speak', methods=['POST'])
def speak():
    text = request.form['text']
    language = request.form['language']
    
    audio_file = generate_audio(text, language)
    
    return jsonify({'audio_file': audio_file})

if __name__ == "__main__":
    app.run(debug=True)

