from django.shortcuts import render, redirect
import numpy as np
import cv2
import PIL.Image
from google import genai
from huggingface_hub import InferenceClient
from requests.exceptions import HTTPError
from .models import Upload
from django.core.files import File
import os
from uuid import uuid4
import random
from google.genai.errors import ClientError, ServerError

# ---------------- GEMINI CLIENT ----------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------- HUGGINGFACE CLIENT ----------------
hf_token = os.getenv("HF_TOKEN")
image_client = InferenceClient(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    token=hf_token
)

# ---------------- FACE DETECTION ----------------
def is_human_face(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return len(faces) > 0

# ---------------- IMAGE UPLOAD VIEW ----------------
def upload_image(request):
    if request.method == 'POST':
        gender = request.POST.get('gender')
        style_type = request.POST.get('styleType')
        clothing_category = request.POST.get('clothingCategory')
        user_image = request.FILES.get('userImage')

        if user_image and is_human_face(user_image):
            user_image.seek(0)

            if style_type == "Clothes":
                return generate_clothes_images(request, user_image, gender, clothing_category)

            elif style_type == "Glasses":
                return generate_glasses_images(request, user_image, gender)

            else:
                print("Invalid style type selected.")
                return render(request, 'Upload.html', {'message': 'Invalid style type selected.'})
        else:
            print("No face detected or invalid image.")
            return render(request, 'Upload.html', {'message': 'No face detected or invalid image.'})

    return render(request, 'Upload.html')

# ---------------- CLOTHES GENERATION ----------------
def generate_clothes_images(request, user_image, gender, category):
    image = PIL.Image.open(user_image)

    for _ in range(2):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    "Analyze the uploaded image to detect the person's gender and skin tone.",
                    "Generate a unique clothing design prompt for an image generation model.",
                    f"Ensure the clothing matches {gender} style and fits into the '{category}' category.",
                    "Use colors that complement the person's skin tone and avoid repeating past styles.",
                    "Do not include any human or background — only the clothing item.",
                    image
                ]
            )
            # Only access response if it is valid
            if response and hasattr(response, 'text') and response.text:
                generated_prompt = response.text.strip()
                print("🧥 Clothing Prompt:", generated_prompt)
                image_generation(generated_prompt)
            else:
                print("Gemini returned no text.")
                return render(request, 'Upload.html', {'message': 'Gemini returned no output. Please try again.'})

        except (ClientError, ServerError) as e:
            print("Gemini API error:", e)
            return render(request, 'Upload.html', {'message': 'Gemini model is busy or unavailable. Please try again later.'})
        except Exception as e:
            print("Unexpected error:", e)
            return render(request, 'Upload.html', {'message': 'An unexpected error occurred. Please try again.'})

    return redirect('display_image')

# ---------------- GLASSES GENERATION ----------------
def generate_glasses_images(request, user_image, gender):
    image = PIL.Image.open(user_image)

    style_keywords = [
        "retro style", "futuristic", "bold frames", "minimalist",
        "luxury brand style", "colorful acetate", "transparent frames",
        "aviator style", "round vintage", "fashion-forward design"
    ]

    for _ in range(2):
        random_style = random.choice(style_keywords)
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    "Analyze the uploaded image to detect the person's skin tone and face shape.",
                    "Generate a unique glasses design prompt for an image generation model.",
                    f"The glasses should match {gender} fashion and suit the detected face shape and skin tone.",
                    f"Use a {random_style} in the design.",
                    "Only generate the glasses on a transparent or plain background — do not include a human face.",
                    image
                ]
            )
            # Only access response if it is valid
            if response and hasattr(response, 'text') and response.text:
                generated_prompt = response.text.strip()
                print("🕶️ Glasses Prompt:", generated_prompt)
                image_generation(generated_prompt)
            else:
                print("Gemini returned no text.")
                return render(request, 'Upload.html', {'message': 'Gemini returned no output. Please try again.'})

        except (ClientError, ServerError) as e:
            print("Gemini API error:", e)
            return render(request, 'Upload.html', {'message': 'Gemini model is busy or unavailable. Please try again later.'})
        except Exception as e:
            print("Unexpected error:", e)
            return render(request, 'Upload.html', {'message': 'An unexpected error occurred. Please try again.'})

    return redirect('display_image')

# ---------------- IMAGE GENERATION (STABLE DIFFUSION) ----------------
def image_generation(prompt):
    try:
        generated_image = image_client.text_to_image(prompt)
    except HTTPError as e:
        print("HuggingFace API HTTP error:", e)
        raise Exception("Stable Diffusion service unavailable. Please try again later.")
    except Exception as e:
        print("Unexpected error during image generation:", e)
        raise Exception("Unexpected error during image generation.")

    unique_filename = f"generatedimage_{uuid4().hex}.png"
    path = os.path.join("media/uploads", unique_filename)

    generated_image.save(path, format="PNG")

    with open(path, "rb") as f:
        django_file = File(f)
        Upload.objects.create(userImage=django_file)

# ---------------- DISPLAY IMAGES ----------------
def display_image(request):
    images = Upload.objects.all().order_by('-created_at')[:2]
    return render(request, 'showImages.html', {'images': images})

def see_image(request):
    images = list(Upload.objects.all())
    images = random.choices(images, k=10)
    return render(request, 'showImages.html', {'images': images})
