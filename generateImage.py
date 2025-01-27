import random
import os
from datetime import datetime
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline
from google.colab import drive
import shutil
import threading

prompts = [
    "A glowing cyberpunk cityscape at night with neon lights reflecting in puddles, detailed and futuristic.",
    "A surreal fusion of a roaring lion and an erupting volcano, vibrant flames blending into the mane.",
    "An astronaut floating in a galaxy of jellyfish, with vivid cosmic colors and ethereal lighting.",
    "A biomechanical dragon with gears and steam, merging fantasy with industrial steampunk design.",
    "A minimalist mountain range with a giant moon rising, blending pastel and metallic gradients.",
    "A phoenix rising from a vortex of abstract geometric shapes, fiery and energetic.",
    "A retro 80s vaporwave aesthetic with palm trees, sunsets, and gridlines, glowing in neon.",
    "A detailed octopus wearing headphones, DJing on a turntable submerged underwater.",
    "A wolf howling at a shattered moon, with glowing shards illuminating the forest below.",
    "A tree growing out of a human heart, with intricate veins transforming into roots and branches.",
    "A robotic samurai standing in a bamboo forest, with glowing katana and electric accents.",
    "A swirling black hole consuming planets, with vibrant cosmic energy bursting outward.",
    "A fusion of a hummingbird and a flower in motion, with trails of vibrant, liquid paint.",
    "A skeleton riding a bicycle made of flowers, juxtaposing life and death in colorful tones.",
    "A chameleon blending into a graffiti wall, its scales mirroring the bold urban art.",
    "A cosmic whale swimming through a sea of stars, surrounded by glowing constellations.",
    "A time traveler stepping out of a clock-shaped portal, gears and glowing numbers swirling.",
    "A mashup of nature and technology: a deer with antlers made of circuit boards.",
    "A surreal dreamscape of a melting clock, spilling time into a vibrant desert.",
    "A futuristic samurai helmet with neon tribal patterns glowing against a dark backdrop.",
    "A fierce tiger breaking through a cracked mirror, with reflections showing different vibrant realities.",
    "An enchanted forest with glowing mushrooms and fireflies, where a hidden doorway leads to another world.",
    "A pirate ship sailing through stormy clouds, with lightning bolts shaped like skulls.",
    "A galaxy encapsulated in a bottle, with stars, planets, and swirling nebulae pouring out.",
    "A futuristic astronaut riding a motorcycle made of light beams on a glowing alien planet.",
    "A mechanical butterfly with intricate gears and glowing wings, flying over a dystopian cityscape.",
    "A fox made of flames leaping through a frozen forest, leaving trails of fire and ice.",
    "A surreal mashup of a chessboard and a mountain landscape, with chess pieces towering like skyscrapers.",
    "A crystal skull reflecting a kaleidoscope of colors, surrounded by vines and roses made of glass.",
    "A cosmic yin-yang symbol, with one side made of galaxies and stars and the other of lush, vibrant forests."
]

# Load the model
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Use GPU

# Function to generate and save an image
def generate_image(prompt, num_inference_steps=100, guidance_scale=12.5, seed=42, 
                   save_path="generated_images", image_format="PNG", width=512, height=512, contrast=None):
    os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists
    generator = torch.manual_seed(seed)  # Set seed for reproducibility

    # Generate image
    image = pipe(prompt, num_inference_steps=num_inference_steps, 
                 guidance_scale=guidance_scale, generator=generator, 
                 height=height, width=width).images[0]

    # Randomize contrast if not provided
    if contrast is None:
        contrast = random.choice(["low", "medium", "high"])

    # Adjust image contrast
    enhancer = ImageEnhance.Contrast(image)
    contrast_levels = {"low": 0.8, "medium": 1.0, "high": 1.5}
    image = enhancer.enhance(contrast_levels.get(contrast.lower(), 1.0))

    # Display the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    # Create filename
    prompt_words = prompt.split()[:5]  # First 5 words of prompt
    prompt_filename = "_".join(prompt_words).replace(",", "").replace(".", "")
    style_words = [word for word in prompt.split() if word in [
        "street", "minimalist", "retro", "cyberpunk", "geometric", "neon", "sketch", "impressionism", "surrealism",
        "pop", "cubism", "abstract", "vintage", "nature", "futurism"]]
    style_filename = "_".join(style_words).replace(",", "").replace(".", "")
    image_filename = os.path.join(save_path, f"{prompt_filename}_{style_filename}_{contrast}_{width}x{height}.{image_format.lower()}")

    # Save the image
    image.save(image_filename, format=image_format.upper())
    print(f"Image saved at: {image_filename}")

    # Convert PNG to PDF
    convert_png_to_pdf(image_filename, save_path)

    # Upload to Google Drive after both image and PDF are ready
    upload_files_to_google_drive(save_path)

# Function to convert PNG to PDF
def convert_png_to_pdf(png_path, save_path):
    image = Image.open(png_path)
    if image.mode == 'RGBA':  # Convert to RGB if needed
        image = image.convert('RGB')
    pdf_path = png_path.replace(".png", ".pdf")
    image.save(pdf_path, "PDF", resolution=100.0)
    print(f"PDF saved at: {pdf_path}")

# Function to generate images for multiple prompts
def generate_images_for_prompts(prompts, batch_size=5, num_inference_steps=150, guidance_scale=15, 
                                 image_format="PNG", width=512, height=512, save_path="generated_images", contrast=None):
    os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists
    for prompt in prompts:
        print(f"Processing prompt: {prompt}")
        for i in range(batch_size):
            print(f"Generating image {i + 1}/{batch_size} for prompt: {prompt}")

            # Add style variations
            style_variations = [
                "in a street art style", 
                "with a minimalist design", 
                "using retro colors and patterns",
                "in a cyberpunk style", 
                "with abstract geometric shapes", 
                "with bold neon colors",
                "in a hand-drawn sketch style", 
                "in an impressionism style", 
                "with surrealistic elements",
                "in a pop art style", 
                "in a cubist style", 
                "with abstract art influence",
                "with vintage elements", 
                "in a nature-inspired design", 
                "in a futuristic style",
                "with steampunk aesthetics", 
                "in a watercolor paint style", 
                "with metallic textures",
                "in a mosaic tile pattern", 
                "in a tribal art style", 
                "with glowing holographic effects",
                "in a 3D rendered design", 
                "with a grunge texture overlay", 
                "using pastel shades and gradients",
                "with a graffiti-inspired look", 
                "in a sci-fi hologram aesthetic", 
                "in an Art Nouveau style",
                "with optical illusion effects", 
                "in a gothic-inspired design"
            ]

            styles = random.sample(style_variations, random.randint(1, 3))  # Randomly choose styles
            styled_prompt = f"{prompt} {' '.join(styles)}"

            # Generate and save the image
            generate_image(styled_prompt, num_inference_steps=num_inference_steps, 
                           guidance_scale=guidance_scale, image_format=image_format, 
                           width=width, height=height, save_path=save_path, contrast=contrast)

# Function to upload files to Google Drive
def upload_files_to_google_drive(local_folder):
    # Mount Google Drive
    drive.mount('/content/drive')

    # Define the parent folder in Google Drive
    parent_folder = '/content/drive/MyDrive/AI_Generated_Images'
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)  # Create the main folder if it doesn't exist

    # Create subfolders for PDFs and PNGs
    pdf_folder = os.path.join(parent_folder, 'pdf')
    png_folder = os.path.join(parent_folder, 'png')

    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)

    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    # Upload files from the local folder to Google Drive, sorted by type
    for filename in os.listdir(local_folder):
        file_path = os.path.join(local_folder, filename)
        if os.path.isfile(file_path):
            if filename.lower().endswith('.pdf'):
                shutil.copy(file_path, pdf_folder)  # Copy PDF files to Google Drive
            elif filename.lower().endswith('.png'):
                shutil.copy(file_path, png_folder)  # Copy PNG files to Google Drive
    print(f"Files uploaded to Google Drive from {local_folder}")

# Run the image generation for multiple prompts
generate_images_for_prompts(prompts, batch_size=33)
