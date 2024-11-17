import streamlit as st
from transformers import pipeline
import torchaudio
from config import MODEL_ID

# Load the model and pipeline using the model_id variable
pipe = pipeline("audio-classification", model=MODEL_ID)

def classify_audio(filepath):
    preds = pipe(filepath)
    outputs = {"normal": 0.0, "artifact": 0.0, "murmur": 0.0}
    for p in preds:
        label = p["label"]
        # Simplify the labels and accumulate the scores
        if "artifact" in label:
            outputs["artifact"] += p["score"]
        elif "murmur" in label:
            outputs["murmur"] += p["score"]
        elif "extra" in label or "Normal" in label:
            outputs["normal"] += p["score"]
    return outputs

# Streamlit app layout
st.title("Heartbeat Sound Classification")

# Theme selection
theme = st.sidebar.selectbox(
    "Select Theme",
    ["Light Green", "Light Blue"]
)

# Add custom CSS for styling based on the selected theme
if theme == "Light Green":
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #e8f5e9; /* Light green background */
        }
        .stApp {
            color: #004d40; /* Dark green text */
        }
        .stButton > button, .stFileUpload > div {
            background-color: #004d40; /* Dark green button and file uploader background */
            color: white; /* White text */
        }
        .stButton > button:hover, .stFileUpload > div:hover {
            background-color: #00332c; /* Darker green on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
elif theme == "Light Blue":
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #e0f7fa; /* Light blue background */
        }
        .stApp {
            color: #006064; /* Dark blue text */
        }
        .stButton > button, .stFileUpload > div {
            background-color: #006064; /* Dark blue button and file uploader background */
            color: white; /* White text */
        }
        .stButton > button:hover, .stFileUpload > div:hover {
            background-color: #004d40; /* Darker blue on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# File uploader for audio files
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.subheader("Uploaded Audio File")
    # Load and display the audio file
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format='audio/wav')

    # Save the uploaded file to a temporary location
    with open("temp_audio_file.wav", "wb") as f:
        f.write(audio_bytes)
    
    # Classify the audio file
    st.write("Classifying the audio...")
    results = classify_audio("temp_audio_file.wav")
    
    # Display the classification results in a dedicated output box
    st.subheader("Classification Results")
    results_box = st.empty()
    results_str = "\n".join([f"{label}: {score:.2f}" for label, score in results.items()])
    results_box.text(results_str)

# Sample Audio Files for classification
st.write("Sample Audio Files:")
examples = ['normal.wav', 'murmur.wav', 'extra_systole.wav', 'extra_hystole.wav', 'artifact.wav']
for example in examples:
    if st.button(example):
        st.subheader(f"Sample Audio: {example}")
        audio_bytes = open(example, 'rb').read()
        st.audio(audio_bytes, format='audio/wav')
        results = classify_audio(example)
        st.write("Results:")
        results_str = "\n".join([f"{label}: {score:.2f}" for label, score in results.items()])
        st.text(results_str)
