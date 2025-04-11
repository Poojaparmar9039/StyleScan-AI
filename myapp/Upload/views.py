from django.shortcuts import render,redirect
import numpy as np
import cv2
import PIL.Image
import google.generativeai as genai
from huggingface_hub import InferenceClient
from . models import Upload
from django.core.files import File
import os
from uuid import uuid4
import random




def is_human_face(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

def upload_image(request):
    if request.method == 'POST':
        userImage = request.FILES.get('userImage')
        if userImage:
            print("📷 Uploaded Image:", userImage)
            result = is_human_face(userImage)
            if result:
                print("✅ Human face detected!")
                userImage.seek(0)
                return our_image(userImage)
            else:
                print(" No human face detected.")
    return render(request, 'Upload.html')



def our_image(userImage):
    image = PIL.Image.open(userImage)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    for _ in range(2):
        response = model.generate_content([
            "Analyze the uploaded image to detect the person's gender and skin tone. "
            "Based on this analysis, generate a unique prompt for an LLM to create a clothing design that suits the person. "
            "The design should match the person's gender and have a color that complements their skin tone. "
            "Do not repeat colors or styles across different prompts. "
            "The clothing type should vary—choose randomly from modern streetwear, traditional attire, casual outfits, formal wear, suits, or other fashion categories. "
            "Include details about the person's features if relevant and a line that emphasizes exploring new colors. "
            "The final generated image should only include the clothing item—do not include any human or background. "
            "Return only the prompt for image generation, with no extra explanation.",
            image
        ])

        generated_prompt = response.text.strip()
        print("🧠 Generated Prompt:", generated_prompt)
        image_generation(generated_prompt)

    return redirect('display_image')


def image_generation(prompt):
    styles = [
        "in a modern streetwear style",
        "with a minimalistic design",
        "in a vibrant graphic tee style",
        "with a vintage 90s pattern",
        "featuring abstract shapes",
        "with artistic brushstroke textures",
        "traditional wares",
        "with a nature-inspired theme",
        "inspired by pop culture",
        "with a retro vibe",
        "colourful tops",
        "ballon pattern tops"
        "full office wear suit",
        "featuring a futuristic techwear look",
        "with a bold print",
        "in a casual oversized fit"
    ]
    colors=[
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "pink",
        "orange",
        "black",
        "white",
        "grey",
        "brown",
        "beige",
        "teal",
        "navy blue",
        "maroon"
        "turquoise",
        "lavender",
        "coral"
    ]
    style_hint = random.choice(styles)   
    color=random.choice(colors)
    varied_prompt = f"{prompt} The costume should be {style_hint} and of {color}."
    generated_image = image_client.text_to_image(varied_prompt)
    unique_filename = f"generatedimage_{uuid4().hex}.png"
    path = os.path.join("media/uploads", unique_filename)
    generated_image.save(path, format="PNG")
    
    with open(path, "rb") as f:
        django_file = File(f)
        Upload.objects.create(userImage=django_file)

    # return redirect('display_image')
    return redirect('display_image')


def display_image(request):
    images = Upload.objects.all().order_by('-created_at')  
    return render(request, 'showImages.html', {'images': images})

def see_image(request):
    images =Upload.objects.all()
    images=random.choices(images, k=10)
    return render(request, 'showImages.html', {'images': images})