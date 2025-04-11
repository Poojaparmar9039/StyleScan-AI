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


# genai.configure(api_key="")
# hf_token = 

image_client = InferenceClient("stabilityai/stable-diffusion-3-medium-diffusers", token=hf_token)


# def is_human_face(uploaded_file):
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     if img is None:
#         return False
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     return len(faces) > 0

# def upload_image(request):
#     if request.method == 'POST':
#         userImage = request.FILES.get('userImage')
#         if userImage:
#             print("📷 Uploaded Image:", userImage)
#             result = is_human_face(userImage)
#             if result:
#                 print("✅ Human face detected!")
#                 userImage.seek(0)
#                 return our_image(userImage)
#             else:
#                 print(" No human face detected.")
#     return render(request, 'Upload.html')


# def our_image(userImage):
#     image = PIL.Image.open(userImage)
#     model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
#     response = model.generate_content([
#         "Analyze the uploaded image to detect the person's skin tone. "
#         "Based on the analysis, generate a prompt for an LLM to create an image of a t-shirt that suits the person. "
#         "The t-shirt design should be in a random color that complements the skin tone. "
#         "The generated image should not include any human—only the t-shirt based on the user's features. "
#         "Return only the prompt for image generation, without any explanation.",
#         image
#     ])
#     generated_prompt = response.text.strip()
#     print("🧠 Generated Prompt:", generated_prompt)
#     return image_generation(generated_prompt)


# def image_generation(base_prompt):
#     styles = [
#         "in a modern streetwear style",
#         "with a minimalistic design",
#         "in a vibrant graphic tee style",
#         "with a vintage 90s pattern",
#         "featuring abstract shapes",
#         "with artistic brushstroke textures",
#         "in a sporty athleisure style",
#         "featuring a futuristic techwear look",
#         "with a bold print",
#         "in a casual oversized fit"
#     ]

#     for i in range(10):
#         style_hint = random.choice(styles)   
#         varied_prompt = f"{base_prompt} The t-shirt should be {style_hint}."
#         generated_image = image_client.text_to_image(varied_prompt)
#         unique_filename = f"generatedimage_{uuid4().hex}.png"
#         path = os.path.join("media/uploads", unique_filename)
#         generated_image.save(path, format="PNG")
#         with open(path, "rb") as f:
#             django_file = File(f)
#             Upload.objects.create(userImage=django_file)
#     return redirect('display_image')

# def display_image(request):
#     images = Upload.objects.all().order_by('-created_at')  
#     return render(request, 'upload.html', {'images': images})




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
    response = model.generate_content([
        "Analyze the uploaded image to detect the person's skin tone. "
        "Based on the analysis, generate a prompt for an LLM to create an image of a t-shirt that suits the person. "
        "The t-shirt design should be in a random color that complements the skin tone. "
        "dont generate similar color or similar prompt"
        "the prompt should contain features of the person and also a line the it alsways access new colors "
        "The generated image should not include any human—only the t-shirt based on the user's features. "
        "Return only the prompt for image generation, without any explanation.",
        image
    ])
    generated_prompt = response.text.strip()
    print("🧠 Generated Prompt:", generated_prompt)
    return image_generation(generated_prompt)


def image_generation(prompt):
    styles = [
        "in a modern streetwear style",
        "with a minimalistic design",
        "in a vibrant graphic tee style",
        "with a vintage 90s pattern",
        "featuring abstract shapes",
        "with artistic brushstroke textures",
        "in a sporty athleisure style",
        "featuring a futuristic techwear look",
        "with a bold print",
        "in a casual oversized fit"
    ]
    style_hint = random.choice(styles)   
    varied_prompt = f"{prompt} The t-shirt should be {style_hint}."
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

