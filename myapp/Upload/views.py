from django.shortcuts import render, redirect
import numpy as np
import cv2
import PIL.Image
import google.generativeai as genai
from huggingface_hub import InferenceClient
from .models import Upload
from django.core.files import File
import os
from uuid import uuid4
import random



genai.configure(api_key=os.getenv("GENAI_API_KEY"))
hf_token = os.getenv("HF_TOKEN")
image_client = InferenceClient("stabilityai/stable-diffusion-3-medium-diffusers", token=hf_token)


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
        gender = request.POST.get('gender')
        style_type = request.POST.get('styleType')  # "Clothes" or "Glasses"
        clothing_category = request.POST.get('clothingCategory')  # may be None
        user_image = request.FILES.get('userImage')

        if user_image and is_human_face(user_image):
            user_image.seek(0)
            if style_type == "Clothes":
                return generate_clothes_images(user_image, gender, clothing_category)
            elif style_type == "Glasses":
                return generate_glasses_images(user_image, gender)
            else:
                print("Invalid style type selected.")
        else:
            print("No face detected or invalid image.")
            # return render(request, 'Upload.html', {'message': 'No face detected or invalid image.'})
    return render(request, 'Upload.html')


def generate_clothes_images(user_image, gender, category):
    image = PIL.Image.open(user_image)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    for _ in range(2):
        response = model.generate_content([
            f"Analyze the uploaded image to detect the person's gender and skin tone.",
            f"Generate a unique clothing design prompt for an image generation model.",
            f"Ensure the clothing matches {gender} style and fits into the '{category}' category.",
            f"Use colors that complement the person's skin tone and avoid repeating past styles.",
            f"Do not include any human or background — only the clothing item.",
            image
        ])
        generated_prompt = response.text.strip()
        print("🧥 Clothing Prompt:", generated_prompt)
        image_generation(generated_prompt)

    return redirect('display_image')


def generate_glasses_images(user_image, gender):
    image = PIL.Image.open(user_image)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    
    style_keywords = [
        "retro style", "futuristic", "bold frames", "minimalist", "luxury brand style",
        "colorful acetate", "transparent frames", "aviator style", "round vintage", "fashion-forward design"
    ]

    for _ in range(2):
        random_style = random.choice(style_keywords)

        response = model.generate_content([
            f"Analyze the uploaded image to detect the person's skin tone and face shape.",
            f"Generate a unique glasses design prompt for an image generation model.",
            f"The glasses should match {gender} fashion and suit the detected face shape and skin tone.",
            f"Use a {random_style} in the design.",
            f"Only generate the glasses on a transparent or plain background — do not include a human face.",
            image
        ])
        generated_prompt = response.text.strip()
        print("🕶️ Glasses Prompt:", generated_prompt)
        image_generation(generated_prompt)

    return redirect('display_image')



def image_generation(prompt):
    generated_image = image_client.text_to_image(prompt)
    unique_filename = f"generatedimage_{uuid4().hex}.png"
    path = os.path.join("media/uploads", unique_filename)
    generated_image.save(path, format="PNG")

    with open(path, "rb") as f:
        django_file = File(f)
        Upload.objects.create(userImage=django_file)


def display_image(request):
    images = Upload.objects.all().order_by('-created_at')[:2]
    return render(request, 'showImages.html', {'images': images})


def see_image(request):
    images = Upload.objects.all()
    images = random.choices(images, k=10)
    return render(request, 'showImages.html', {'images': images})
