import streamlit as st
import speech_recognition as sr
import pyttsx3
import random
import sqlite3
import time
import spacy
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sentence_transformers import SentenceTransformer

# ====== Step 1: Silence Huggingface Hub Warning ======
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# ====== Step 2: Load spaCy Model ======
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_lg")

nlp = load_spacy_model()

# ====== Step 3: Build Siamese Network for Sentence Similarity ======
def build_siamese_network():
    input_1 = Input(shape=(384,))  # Adjusted for 'all-MiniLM-L6-v2' SentenceTransformer embeddings
    input_2 = Input(shape=(384,))

    # L1 distance layer
    l1_distance = Lambda(lambda embeddings: K.abs(embeddings[0] - embeddings[1]))([input_1, input_2])

    # Dense layers for similarity scoring
    x = Dense(128, activation='relu')(l1_distance)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)

    # Output layer
    similarity = Dense(1, activation='sigmoid')(x)

    siamese_model = Model(inputs=[input_1, input_2], outputs=similarity)
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return siamese_model

# Initialize the Siamese network
siamese_model = build_siamese_network()

# Load pre-trained weights if available
# siamese_model.load_weights('path_to_your_model_weights.h5')

# ====== Step 4: Load Sentence-BERT Model ======
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get sentence embeddings
def get_sentence_embeddings(text):
    return sentence_model.encode(text)

# ====== Step 5: Similarity Calculation with Siamese Network ======
def calculate_similarity_siamese(user_answer, expected_answer):
    user_embedding = get_sentence_embeddings(user_answer).reshape(1, -1)
    expected_embedding = get_sentence_embeddings(expected_answer).reshape(1, -1)

    similarity = siamese_model.predict([user_embedding, expected_embedding])[0][0]
    return similarity

# ====== Step 6: Load Questions from Database ======
def load_questions_from_db(stream, level):
    conn = sqlite3.connect('interview_questions.db')
    c = conn.cursor()
    c.execute('''SELECT question, expected_answer FROM questions WHERE stream = ? AND level = ?''', (stream, level))
    questions = c.fetchall()
    conn.close()
    return questions

# ====== Step 7: Initialize Streamlit App ======
st.title("Mock Interview Assistant")
st.sidebar.header("Interview Settings")
question_source = st.sidebar.radio("Question Source", ("Resume-Based", "Manually Select Stream"))
difficulty_level = st.sidebar.selectbox("Select Difficulty Level", ["Beginner", "Intermediate", "Advanced"])

resume_text = ''
if question_source == "Resume-Based":
    resume_file = st.sidebar.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    if resume_file is not None:
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(resume_file)

        stream = determine_stream_from_resume(resume_text)
        if not stream:
            st.write("Could not detect a stream based on your resume. Please select manually.")
    else:
        stream = None
else:
    stream = st.sidebar.selectbox("Stream", ["Python", "Java", "C++", "JavaScript", "Data Science", "Machine Learning"])

# ====== Step 8: Session State Initialization ======
if 'current_question' not in st.session_state:
    st.session_state['current_question'] = None
    st.session_state['expected_answer'] = None
    st.session_state['question_asked'] = False
    st.session_state['user_answer'] = ""
    st.session_state['feedback'] = ""
    st.session_state['start_time'] = None
    st.session_state['similarity'] = 0
    st.session_state['response_time'] = 0

# ====== Step 9: Generate Question ======
def generate_question_from_db(stream, level):
    questions = load_questions_from_db(stream, level)
    if questions:
        return random.choice(questions)
    else:
        st.error("No questions available for the selected stream and level.")
        return None, None

# ====== Step 10: Ask Question Section ======
st.header("Interview Question")
if st.button("Ask Question"):
    if stream and difficulty_level:
        question_data = generate_question_from_db(stream, difficulty_level)
        if question_data:
            st.session_state['current_question'], st.session_state['expected_answer'] = question_data
            st.session_state['question_asked'] = True
            st.session_state['start_time'] = time.time()
            st.write("### Question:")
            st.write(st.session_state['current_question'])

            engine = pyttsx3.init()
            engine.say(st.session_state['current_question'])
            engine.runAndWait()

# ====== Step 11: Speech Input Section ======
st.header("Speech Input")
if st.session_state['question_asked'] and st.button("Record Speech"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Recording... Please speak clearly.")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.write("Processing...")
            user_answer = recognizer.recognize_google(audio)
            st.session_state['user_answer'] = user_answer
            st.text_area("Your Answer:", st.session_state['user_answer'])

            # Calculate response time
            st.session_state['response_time'] = time.time() - st.session_state['start_time']

            # Calculate similarity using Siamese network
            expected_answer = st.session_state['expected_answer']
            similarity = calculate_similarity_siamese(user_answer, expected_answer)
            st.session_state['similarity'] = similarity

            st.write(f"**Siamese Network Similarity Score:** {similarity * 100:.2f}%")
            st.write("### Feedback: The similarity score indicates how well your answer matches the expected answer.")

        except sr.WaitTimeoutError:
            st.error("Listening timed out. Please try again.")
        except sr.UnknownValueError:
            st.error("Could not understand the audio. Please try again.")
        except sr.RequestError:
            st.error("Speech recognition service is unavailable.")

# ====== Step 12: Performance Metrics Section ======
st.header("Performance Metrics")
col1, col2 = st.columns(2)

with col1:
    st.metric("Similarity Score", f"{st.session_state['similarity'] * 100:.2f}%")
    st.metric("Response Time (s)", f"{st.session_state['response_time']:.2f}")
