import streamlit as st
import speech_recognition as sr
import pyttsx3
import random
import sqlite3
import time
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the spaCy model
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

st.title("Mock Interview Assistant")

# Question Source and Difficulty Level
st.sidebar.header("Interview Settings")
question_source = st.sidebar.radio(
    "Would you like to be asked questions based on your resume or manually select a stream?",
    ("Resume-Based", "Manually Select Stream")
)

difficulty_level = st.sidebar.selectbox("Select Difficulty Level", ["Beginner", "Intermediate", "Advanced"])

# Add new streams like Data Science and Machine Learning
if question_source == "Manually Select Stream":
    stream = st.sidebar.selectbox("Stream", ["Python", "Java", "C++", "JavaScript", "Data Science", "Machine Learning"])
else:
    stream = None  # Placeholder for resume-based stream

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

            # Calculate similarity between user answer and expected answer using spaCy
            st.session_state['similarity'] = nlp(user_answer).similarity(nlp(expected_answer))
            st.write(f"Similarity Score: {st.session_state['similarity'] * 100:.2f}%")

            # Calculate Word Error Rate (WER)
            st.session_state['wer'] = calculate_wer(expected_answer.split(), user_answer.split())
            
            # Keywords for evaluation
            keywords = ["keyword1", "keyword2", "keyword3"]  # Customize for your domain

            # Keyword-based precision, recall, F1-score
            y_true = [1 if kw in expected_answer else 0 for kw in keywords]
            y_pred = [1 if kw in user_answer else 0 for kw in keywords]
            st.session_state['precision'] = precision_score(y_true, y_pred, zero_division=1)
            st.session_state['recall'] = recall_score(y_true, y_pred, zero_division=1)
            st.session_state['f1_score'] = f1_score(y_true, y_pred, zero_division=1)

            # Provide feedback
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
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Similarity Score", value=f"{st.session_state['similarity'] * 100:.2f}%")
    st.metric(label="Precision", value=f"{st.session_state['precision']:.2f}")
    
with col2:
    st.metric(label="Response Time", value=f"{st.session_state['response_time']:.2f} sec")
    st.metric(label="Recall", value=f"{st.session_state['recall']:.2f}")
    
with col3:
    st.metric(label="Word Error Rate", value=f"{st.session_state['wer']:.2f}")
    st.metric(label="F1 Score", value=f"{st.session_state['f1_score']:.2f}")
