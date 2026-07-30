import ollama
import speech_recognition as sr
import pyttsx3

# Initialize TTS
engine = pyttsx3.init()

def speak(text):
    print("Bot:", text)
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)

            print("You:", text)
            return text

        except sr.UnknownValueError:
            print("Could not understand audio.")
            return ""

        except Exception as e:
            print(e)
            return ""

def chat():
    print("===== Voice Chatbot =====")
    print("Say 'exit' to quit.")

    while True:

        user_input = listen()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            speak("Goodbye")
            break

        try:
            response = ollama.chat(
                model="qwen3:latest",
                messages=[
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            )

            answer = response["message"]["content"]

            speak(answer)

        except Exception as e:
            print("Error:", e)
            speak("Cannot connect to Ollama.")

if __name__ == "__main__":
    chat()