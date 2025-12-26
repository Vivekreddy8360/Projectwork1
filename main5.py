import streamlit as st
import speech_recognition as sr
import pyttsx3
import random
import sqlite3
import time
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score
from PyPDF2 import PdfReader
import docx

# Load the larger spaCy model with word vectors for more accurate similarity
nlp = spacy.load("en_core_web_lg")

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
    resume_file = st.sidebar.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    if resume_file is not None:
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(resume_file)

        st.write("Extracted Resume Text:")
        st.write(resume_text)

        stream = determine_stream_from_resume(resume_text)
        if stream:
            st.write(f"Detected stream: {stream}")
        else:
            st.write("Could not detect a stream based on your resume. Please select manually.")
    else:
        stream = None
else:
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
    st.session_state['similarity'] = 0
    st.session_state['response_time'] = 0
    st.session_state['precision'] = 0
    st.session_state['recall'] = 0
    st.session_state['f1_score'] = 0
    st.session_state['wer'] = 0

# Function to randomly select a question from the database
def generate_question_from_db(stream, level):
    questions = load_questions_from_db(stream, level)  # Load questions from SQLite DB
    if questions:
        return random.choice(questions)
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
        if question_data:
            st.session_state['current_question'], st.session_state['expected_answer'] = question_data
            st.session_state['question_asked'] = True
            st.session_state['start_time'] = time.time()
            st.write("Question:", st.session_state['current_question'])
            
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

            st.session_state['response_time'] = time.time() - st.session_state['start_time']

            expected_answer = st.session_state['expected_answer']
            user_answer = st.session_state['user_answer']

            st.session_state['similarity'] = nlp(user_answer).similarity(nlp(expected_answer))
            st.write(f"Similarity Score: {st.session_state['similarity'] * 100:.2f}%")

            st.session_state['wer'] = calculate_wer(expected_answer.split(), user_answer.split())
            
            keywords = ["keyword1", "keyword2", "keyword3"]

            y_true = [1 if kw in expected_answer else 0 for kw in keywords]
            y_pred = [1 if kw in user_answer else 0 for kw in keywords]
            st.session_state['precision'] = precision_score(y_true, y_pred, zero_division=1)
            st.session_state['recall'] = recall_score(y_true, y_pred, zero_division=1)
            st.session_state['f1_score'] = f1_score(y_true, y_pred, zero_division=1)

            if st.session_state['similarity'] > 0.75:
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

# Display performance metrics in a box-like structure
st.header("Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Response Time (s)", f"{st.session_state['response_time']:.2f}")

with col2:
    st.metric("WER", f"{st.session_state['wer']:.2f}")

with col3:
    st.metric("Precision", f"{st.session_state['precision'] * 100:.2f}%")

with col4:
    st.metric("Recall", f"{st.session_state['recall'] * 100:.2f}%")

# Function to reset session state for the next question
def reset_for_next_question():
    st.session_state['current_question'] = None
    st.session_state['expected_answer'] = None
    st.session_state['question_asked'] = False
    st.session_state['user_answer'] = ""
    st.session_state['feedback'] = ""
    st.session_state['start_time'] = None
    st.session_state['similarity'] = 0
    st.session_state['response_time'] = 0
    st.session_state['precision'] = 0
    st.session_state['recall'] = 0
    st.session_state['f1_score'] = 0
    st.session_state['wer'] = 0

# Button to move to the next question
if st.button("Next Question"):
    reset_for_next_question()

# Display button for manual level change
if st.button("Change Level"):
    reset_for_next_question()
