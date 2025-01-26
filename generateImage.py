import random
import os
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline

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
    prompt_words = prompt.split()[:5]  # Get the first 5 words of the prompt for the filename
    prompt_filename = "_".join(prompt_words).replace(",", "").replace(".", "")  # Clean the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = os.path.join(save_path, f"{prompt_filename}_{timestamp}_{width}x{height}.{image_format.lower()}")
    
    # Save as PNG, JPEG, or any other supported format
    image.save(image_filename, format=image_format.upper())
    
    print(f"Image saved at: {image_filename}")

# Function to generate a batch of images with style variations
def generate_batch_images(prompt, batch_size=5, num_inference_steps=100, guidance_scale=15, 
                          image_format="PNG", width=512, height=512, save_path="generated_images"):
    """
    Generates and saves a batch of images with different styles for the same prompt.
    
    :param prompt: Text prompt for generating the image
    :param batch_size: Number of images to generate
    :param num_inference_steps: Number of inference steps (higher for better quality)
    :param guidance_scale: Guidance scale for better adherence to the prompt
    :param image_format: Format to save the image (PNG, JPEG)
    :param width: Width of the generated image
    :param height: Height of the generated image
    :param save_path: Folder path to save the generated images
    """
    for i in range(batch_size):
        print(f"Generating image {i + 1}/{batch_size}...")
        
        # Add style variations for diversity
        style_variations = [
            "in a street art style",
            "with a minimalist design",
            "using retro colors and patterns",
            "in a cyberpunk style",
            "with abstract geometric shapes",
            "with bold neon colors",
            "in a hand-drawn sketch style"
        ]
        
        # Randomly choose one or more styles for each batch
        styles = random.sample(style_variations, random.randint(1, 3))  # Choose 1 to 3 styles
        styled_prompt = f"{prompt} {' '.join(styles)}"
        
        # Generate and save the image
        generate_image(styled_prompt, num_inference_steps=num_inference_steps, 
                       guidance_scale=guidance_scale, image_format=image_format, 
                       width=width, height=height, save_path=save_path)

# Example: Generate 5 trendy t-shirt designs with different styles
generate_batch_images("A cool, trendy t-shirt design with bold colors and modern graphic elements", 
                      batch_size=5, num_inference_steps=100, guidance_scale=15, 
                      image_format="PNG", width=512, height=512)
