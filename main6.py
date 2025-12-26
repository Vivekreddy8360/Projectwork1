import streamlit as st
import speech_recognition as sr
import pyttsx3
import random
import sqlite3
import time
import spacy
import gensim
from sklearn.metrics import precision_score, recall_score, f1_score
from PyPDF2 import PdfReader
import docx
import numpy as np

# Load spaCy model for similarity
nlp = spacy.load("en_core_web_lg")

# Load or train a pre-trained CBOW model using Gensim (You can train with your dataset or use a pre-trained one)
model = gensim.models.Word2Vec.load("pretrained_cbow.model")  # Make sure to have a CBOW model loaded

# Function to connect to SQLite database and retrieve questions
def load_questions_from_db(stream, level):
    conn = sqlite3.connect('interview_questions.db')  # Connect to the SQLite DB
    c = conn.cursor()

    # Query to select questions based on stream and level
    c.execute('''
        SELECT question, expected_answer 
        FROM questions 
        WHERE stream = ? AND level = ?
    ''', (stream, level))

    questions = c.fetchall()  # Fetch all matching rows
    conn.close()

    return questions

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to determine the stream based on the resume content
def determine_stream_from_resume(resume_text):
    # You can customize this with keywords to match streams (basic example)
    if 'Python' in resume_text:
        return 'Python'
    elif 'Java' in resume_text:
        return 'Java'
    elif 'Machine Learning' in resume_text:
        return 'Machine Learning'
    elif 'Data Science' in resume_text:
        return 'Data Science'
    else:
        return None

# Function to calculate the similarity using CBOW model
def cbow_similarity(user_answer, expected_answer):
    user_vec = np.mean([model.wv[word] for word in user_answer.lower().split() if word in model.wv], axis=0)
    expected_vec = np.mean([model.wv[word] for word in expected_answer.lower().split() if word in model.wv], axis=0)
    
    # Cosine similarity
    sim = np.dot(user_vec, expected_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(expected_vec))
    return sim

st.title("Mock Interview Assistant")

# Question Source and Difficulty Level
st.sidebar.header("Interview Settings")
question_source = st.sidebar.radio(
    "Would you like to be asked questions based on your resume or manually select a stream?",
    ("Resume-Based", "Manually Select Stream")
)

difficulty_level = st.sidebar.selectbox("Select Difficulty Level", ["Beginner", "Intermediate", "Advanced"])

resume_text = ''
if question_source == "Resume-Based":
    # File uploader for resume (PDF or DOCX)
    resume_file = st.sidebar.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    if resume_file is not None:
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(resume_file)

        # Display extracted resume text
        st.write("Extracted Resume Text:")
        st.write(resume_text)

        # Determine stream based on the resume content
        stream = determine_stream_from_resume(resume_text)
        if stream:
            st.write(f"Detected stream: {stream}")
        else:
            st.write("Could not detect a stream based on your resume. Please select manually.")
    else:
        stream = None
else:
    # Manually select the stream
    stream = st.sidebar.selectbox("Stream", ["Python", "Java", "C++", "JavaScript", "Data Science", "Machine Learning"])

st.write(f"Welcome to the Mock Interview Assistant. Prepare for your interview by answering questions.")

# Initialize session state
if 'current_question' not in st.session_state:
    st.session_state['current_question'] = None
    st.session_state['expected_answer'] = None
    st.session_state['question_asked'] = False
    st.session_state['user_answer'] = ""
    st.session_state['feedback'] = ""
    st.session_state['start_time'] = None
    st.session_state['similarity'] = []
    st.session_state['response_time'] = 0
    st.session_state['precision'] = []
    st.session_state['recall'] = []
    st.session_state['f1_score'] = []
    st.session_state['wer'] = 0
    st.session_state['average_similarity'] = 0

# Function to randomly select a question and its expected answer from the database
def generate_question_from_db(stream, level):
    questions = load_questions_from_db(stream, level)  # Load questions from SQLite DB
    if questions:
        return random.choice(questions)  # Randomly select a question and its expected answer
    else:
        st.error("No questions available for the selected stream and level.")
        return None, None

# Function to calculate Word Error Rate (WER)
def calculate_wer(reference, hypothesis):
    d = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(1, len(reference) + 1):
        d[i][0] = i
    for j in range(1, len(hypothesis) + 1):
        d[0][j] = j
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i-1] == hypothesis[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    wer = d[-1][-1] / len(reference)
    return wer

# Ask the question via voice
st.header("Interview Question")
if st.button("Ask Question"):
    if stream and difficulty_level:
        question_data = generate_question_from_db(stream, difficulty_level)
        if question_data:  # Ensure the question is generated successfully
            st.session_state['current_question'], st.session_state['expected_answer'] = question_data
            st.session_state['question_asked'] = True
            st.session_state['start_time'] = time.time()  # Start timer for response time
            st.write("Question:", st.session_state['current_question'])
            
            # Reinitialize the text-to-speech engine each time
            engine = pyttsx3.init()
            engine.say(st.session_state['current_question'])
            try:
                engine.runAndWait()
            except RuntimeError:
                pass  # Suppress the error message about the loop already running

# Initialize the speech recognizer
recognizer = sr.Recognizer()

st.header("Speech Input")
if st.session_state['question_asked'] and st.button("Record Speech"):
    with sr.Microphone() as source:
        st.write("Recording... Please speak clearly.")
        try:
            audio = recognizer.listen(source, timeout=5)  # Timeout after 5 seconds of silence
            st.write("Processing...")
            st.session_state['user_answer'] = recognizer.recognize_google(audio)
            st.text_area("Your Answer:", st.session_state['user_answer'])

            # Calculate response time
            st.session_state['response_time'] = time.time() - st.session_state['start_time']

            # Compare the user's answer with the expected answer from the database
            expected_answer = st.session_state['expected_answer']
            user_answer = st.session_state['user_answer']

            # Calculate similarity using CBOW model
            cbow_sim = cbow_similarity(user_answer, expected_answer)
            st.session_state['similarity'].append(cbow_sim)
            st.write(f"CBOW Similarity Score: {cbow_sim * 100:.2f}%")

            # Calculate Word Error Rate (WER)
            st.session_state['wer'] = calculate_wer(expected_answer.split(), user_answer.split())

            # Keywords for evaluation
            keywords = ["keyword1", "keyword2", "keyword3"]  # Customize for your domain

            # Keyword-based precision, recall, F1-score
            y_true = [1 if kw in expected_answer else 0 for kw in keywords]
            y_pred = [1 if kw in user_answer else 0 for kw in keywords]
            precision = precision_score(y_true, y_pred, zero_division=1)
            recall = recall_score(y_true, y_pred, zero_division=1)
            f1 = f1_score(y_true, y_pred, zero_division=1)

            st.session_state['precision'].append(precision)
            st.session_state['recall'].append(recall)
            st.session_state['f1_score'].append(f1)

            # Provide feedback based on similarity
            if cbow_sim > 0.75:
                st.session_state['feedback'] = f"Good answer! Your response closely matches the expected answer."
            else:
                st.session_state['feedback'] = f"Your answer could be improved. Try including more relevant details."

            st.write("Feedback:", st.session_state['feedback'])

        except sr.UnknownValueError:
            st.write("Could not understand the audio, please try again.")
        except sr.RequestError:
            st.write("Speech recognition service is unavailable, please check your connection.")
        except TimeoutError:
            st.write("Timeout occurred, please speak within the time limit.")

# Display performance metrics and track progression
st.header("Performance Metrics")
col1, col2, col3 = st.columns(3)

average_similarity = np.mean(st.session_state['similarity']) if st.session_state['similarity'] else 0

with col1:
    st.metric(label="Average Similarity Score", value=f"{average_similarity * 100:.2f}%")
    st.metric(label="Precision", value=f"{np.mean(st.session_state['precision']):.2f}")

with col2:
    st.metric(label="Response Time", value=f"{st.session_state['response_time']:.2f} sec")
    st.metric(label="Recall", value=f"{np.mean(st.session_state['recall']):.2f}")

with col3:
    st.metric(label="Word Error Rate", value=f"{st.session_state['wer']:.2f}")
    st.metric(label="F1 Score", value=f"{np.mean(st.session_state['f1_score']):.2f}")

# Check if the user qualifies to move to the next difficulty level
if average_similarity > 0.75 and len(st.session_state['similarity']) >= 3:
    st.write("Congratulations! You've performed well enough to move to the next level.")
    st.balloons()  # Show a pop-up message or visual indication
    # Reset or proceed to the next level
    st.session_state['similarity'] = []
    st.session_state['precision'] = []
    st.session_state['recall'] = []
    st.session_state['f1_score'] = []
    st.session_state['wer'] = 0
    # Switch to the next difficulty level if needed

