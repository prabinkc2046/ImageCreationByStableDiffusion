!pip install diffusers transformers accelerate torch
!pip install git+https://github.com/huggingface/diffusers

import torch
torch.cuda.is_available()

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Authentication to Hugging Face (if you need it)
# from huggingface_hub import login
# login(token='YOUR_HUGGING_FACE_TOKEN')

# Load the model (use "stabilityai/stable-diffusion-2-1-base" as an example)
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Move the model to GPU
pipe = pipe.to("cuda")  # Ensure you're using GPU

# Function to generate and save the image with custom prompt, settings, and size
def generate_image(prompt, num_inference_steps=75, guidance_scale=12.5, seed=42, 
                   save_path="generated_images", image_format="PNG", width=512, height=512):
    """
    Generates and saves an image based on the provided prompt and size.
    
    :param prompt: Text prompt for generating the image
    :param num_inference_steps: Number of inference steps (higher for better quality)
    :param guidance_scale: Guidance scale for better adherence to the prompt
    :param seed: Seed for reproducibility
    :param save_path: Folder path to save the generated image
    :param image_format: Format to save the image (PNG, JPEG)
    :param width: Width of the generated image
    :param height: Height of the generated image
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Set the seed for reproducibility
    generator = torch.manual_seed(seed)
    
    # Generate the image with custom size
    image = pipe(prompt, num_inference_steps=num_inference_steps, 
                 guidance_scale=guidance_scale, generator=generator, 
                 height=height, width=width).images[0]
    
    # Display the generated image
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    # Save the image locally with a timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = os.path.join(save_path, f"generated_image_{timestamp}_{width}x{height}.{image_format.lower()}")
    
    # Save as PNG, JPEG, or any other supported format
    image.save(image_filename, format=image_format.upper())
    
    print(f"Image saved at: {image_filename}")

# Example 1: Small image (200x200) in PNG format
generate_image("A beautiful sunset over the mountains", num_inference_steps=100, guidance_scale=15, image_format="PNG", width=200, height=200)

# Example 2: Large image (1024x512) in JPEG format
generate_image("A futuristic city skyline with flying cars", num_inference_steps=150, guidance_scale=18, image_format="JPEG", width=1024, height=512)

# Example 3: Default square image (512x512) in PNG format
generate_image("A serene beach", num_inference_steps=75, guidance_scale=12.5, image_format="PNG", width=512, height=512)
