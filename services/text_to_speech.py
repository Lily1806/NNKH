from gtts import gTTS
import tempfile
import os

def speak_text(text, language='vi'):
    """
    Converts text to speech using gTTS and returns the path to the generated MP3 file.
    This approach plays well with Streamlit's st.audio components.
    """
    try:
        # Create a temp file to hold audio
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "speech_output.mp3")
        
        # Generate speech
        tts = gTTS(text=text, lang=language)
        tts.save(temp_path)
        
        return temp_path
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None
