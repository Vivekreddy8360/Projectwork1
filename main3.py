import streamlit as st
import speech_recognition as sr
import pyttsx3
import random
import time
import docx
import pdfplumber
import json

# Load the spaCy model
import spacy
nlp = spacy.load("en_core_web_sm")

# Load questions from a JSON file
def load_questions():
    try:
        with open('questions.json', 'r') as file:
            questions = json.load(file)
        return questions
    except FileNotFoundError:
        st.error("Questions file not found. Please make sure 'questions.json' is in the correct path.")
        return {}

questions = load_questions()

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
    st.session_state['expected_keywords'] = []
    st.session_state['question_asked'] = False
    st.session_state['user_answer'] = ""
    st.session_state['feedback'] = ""

# Function to randomly select a question and its expected keywords
def generate_question(stream, level):
    if stream is None or level is None:
        st.error("Please select a valid stream and difficulty level.")
        return None, []
    
    if stream in questions and level in questions[stream]:
        return random.choice(questions[stream][level])
    else:
        st.error("No questions available for the selected stream and level.")
        return None, []

# Function to evaluate the user's answer by checking keyword matches
def evaluate_answer(user_answer, expected_keywords):
    matched_keywords = [kw for kw in expected_keywords if kw.lower() in user_answer.lower()]
    total_keywords = len(expected_keywords)
    matched_count = len(matched_keywords)
    
    # Calculate a simple match score as a percentage
    score = (matched_count / total_keywords) if total_keywords > 0 else 0
    return score, matched_keywords

# Ask the question via voice
st.header("Interview Question")
if st.button("Ask Question"):
    if stream and difficulty_level:
        question_data = generate_question(stream, difficulty_level)
        if question_data:  # Ensure the question is generated successfully
            st.session_state['current_question'] = question_data['question']
            st.session_state['expected_keywords'] = question_data['expected_keywords']
            st.session_state['question_asked'] = True
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

            # Call the function to evaluate the answer by keyword matching
            score, matched_keywords = evaluate_answer(st.session_state['user_answer'], st.session_state['expected_keywords'])
            st.write(f"Evaluation Score: {score * 100:.2f}%")
            st.write(f"Matched Keywords: {', '.join(matched_keywords)}")

            if score > 0.75:
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
