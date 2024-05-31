from gradio_client import Client  # Import the Client class from the gradio_client module
import pyttsx3
import colorama
import cv2
#import save_pic_webcam

cap = cv2.VideoCapture(2)

colorama.init(autoreset=True)

client = Client("Recognito/FaceAnalysis")  # Create a Client object and connect to the specified Gradio app ("Recognito/FaceAnalysis")

face1_path = r"jee.jpg"
face2_path = r"jee.jpg"


result = client.predict(  # Make a prediction using the Gradio Client, comparing the two specified face images
    face1_path,  # Path to the first face image
    face2_path,  # Path to the second face image
    api_name="/compare_face"  # Specify the API endpoint ("/compare_face") for face comparison
)

print(result)  # Print the result obtained from the Gradio Client prediction

def speak(text):
    engine = pyttsx3.init()
    Id= 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0'
    
    engine.setProperty('voice', Id)
    engine.say(text=text)
    engine.runAndWait()

def format_output(output):
    # Extract the image attributes and matching score from the output
    image1 = output[0]["image1"]["attribute"]  # Extract attributes of the first image from the output
    image2 = output[0]["image2"]["attribute"]  # Extract attributes of the second image from the output
    score = output[0]["matching_score"]  # Extract the matching score from the output

    # Print the matching score and the result
    print(f"Matching score: {score:.2f}")  # Print the matching score with two decimal places
    if score > 0.5:
        print("SAME PERSON")
    else:
        print("DIFFERENT PERSON")

    if score > 0.85:
        speak("Target found!!")

    # Print the attributes of each image in a table
    print("\n\t\tImage 1\t\tImage 2")  # Print header for the table
    print("-" * 100)  # Print a line of dashes for separation
    for key in image1:
        print(f"{key}:\t\t{image1[key]}\t|\t{image2[key]}")  # Print attributes of each image side by side in a table format

format_output(result)  # Call the format_output function to display and format the result obtained from the Gradio Client prediction