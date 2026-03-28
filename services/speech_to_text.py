import speech_recognition as sr

def listen_and_recognize(timeout=5, phrase_time_limit=10, language='vi-VN'):
    """
    Listens to the microphone and returns the recognized text.
    """
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Adjusting microphone for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... (Speak Vietnamese)")
        
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            print("Processing voice...")
            text = recognizer.recognize_google(audio, language=language)
            return text
        except sr.WaitTimeoutError:
            print("No sound heard.")
            return None
        except sr.UnknownValueError:
            print("Could not understand voice.")
            return None
        except sr.RequestError as e:
            print(f"Service connection error: {e}")
            return None
        except Exception as e:
            print(f"Unknown error: {e}")
            return None
