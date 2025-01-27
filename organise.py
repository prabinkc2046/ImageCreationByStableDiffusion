import os
import shutil

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Paths to the folders
base_path = "/content/drive/MyDrive/AI_Generated_Images"
pdf_folder = os.path.join(base_path, "pdf")
png_folder = os.path.join(base_path, "png")

# List of prompts and descriptive folder mappings
descriptive_folders = {
    # "A glowing cyberpunk cityscape at night with neon lights reflecting in puddles, detailed and futuristic.": "Cyberpunk_Cityscape",
    # "A surreal fusion of a roaring lion and an erupting volcano, vibrant flames blending into the mane.": "Lion_Volcano_Fusion",
    # "An astronaut floating in a galaxy of jellyfish, with vivid cosmic colors and ethereal lighting.": "Astronaut_Jellyfish_Galaxy",
    # "A biomechanical dragon with gears and steam, merging fantasy with industrial steampunk design.": "Steampunk_Dragon",
    # "A minimalist mountain range with a giant moon rising, blending pastel and metallic gradients.": "Minimalist_Mountain_Moon",
    "A phoenix rising from a vortex of abstract geometric shapes, fiery and energetic.": "Phoenix_Rising_Art",
    "A retro 80s vaporwave aesthetic with palm trees, sunsets, and gridlines, glowing in neon.": "Retro_Vaporwave",
    "A detailed octopus wearing headphones, DJing on a turntable submerged underwater.": "Octopus_DJ_Underwater",
    "A wolf howling at a shattered moon, with glowing shards illuminating the forest below.": "Wolf_Shattered_Moon",
    "A tree growing out of a human heart, with intricate veins transforming into roots and branches.": "Tree_From_Heart",
    "A robotic samurai standing in a bamboo forest, with glowing katana and electric accents.": "Robotic_Samurai_Bamboo",
    "A swirling black hole consuming planets, with vibrant cosmic energy bursting outward.": "Black_Hole_Cosmic_Energy",
    "A fusion of a hummingbird and a flower in motion, with trails of vibrant, liquid paint.": "Hummingbird_Flower_Fusion",
    "A skeleton riding a bicycle made of flowers, juxtaposing life and death in colorful tones.": "Skeleton_Bicycle_Flowers",
    "A chameleon blending into a graffiti wall, its scales mirroring the bold urban art.": "Chameleon_Graffiti",
    "A cosmic whale swimming through a sea of stars, surrounded by glowing constellations.": "Cosmic_Whale_Stars",
    "A time traveler stepping out of a clock-shaped portal, gears and glowing numbers swirling.": "Time_Traveler_Clock_Portal",
    "A mashup of nature and technology: a deer with antlers made of circuit boards.": "Deer_Circuit_Boards",
    "A surreal dreamscape of a melting clock, spilling time into a vibrant desert.": "Melting_Clock_Dreamscape",
    "A futuristic samurai helmet with neon tribal patterns glowing against a dark backdrop.": "Futuristic_Samurai_Helmet",
    "A fierce tiger breaking through a cracked mirror, with reflections showing different vibrant realities.": "Tiger_Cracked_Mirror",
    "An enchanted forest with glowing mushrooms and fireflies, where a hidden doorway leads to another world.": "Enchanted_Forest_Doorway",
    "A pirate ship sailing through stormy clouds, with lightning bolts shaped like skulls.": "Pirate_Ship_Storm",
    "A galaxy encapsulated in a bottle, with stars, planets, and swirling nebulae pouring out.": "Galaxy_In_Bottle",
    "A futuristic astronaut riding a motorcycle made of light beams on a glowing alien planet.": "Astronaut_Motorcycle_Light",
    "A mechanical butterfly with intricate gears and glowing wings, flying over a dystopian cityscape.": "Mechanical_Butterfly_Cityscape",
    "A fox made of flames leaping through a frozen forest, leaving trails of fire and ice.": "Flaming_Fox_Forest",
    "A surreal mashup of a chessboard and a mountain landscape, with chess pieces towering like skyscrapers.": "Chessboard_Mountain_Mashup",
    "A crystal skull reflecting a kaleidoscope of colors, surrounded by vines and roses made of glass.": "Crystal_Skull_Kaleidoscope",
    "A cosmic yin-yang symbol, with one side made of galaxies and stars and the other of lush, vibrant forests.": "Cosmic_YinYang"
}

# Process each prompt and organize files
for prompt, folder_name in descriptive_folders.items():
    # Get the first five words of the prompt
    prompt_words = prompt.split()[:5]
    prompt_filename = "_".join(prompt_words).replace(",", "").replace(".", "")

    # Create subfolders in pdf and png if they don't exist
    pdf_subfolder = os.path.join(pdf_folder, folder_name)
    png_subfolder = os.path.join(png_folder, folder_name)
    os.makedirs(pdf_subfolder, exist_ok=True)
    os.makedirs(png_subfolder, exist_ok=True)

    # Look for files matching the prompt in the pdf and png folders
    for folder_path, subfolder_path in [(pdf_folder, pdf_subfolder), (png_folder, png_subfolder)]:
        for file_name in os.listdir(folder_path):
            if file_name.startswith(prompt_filename):
                # Move the file to the corresponding subfolder
                src = os.path.join(folder_path, file_name)
                dest = os.path.join(subfolder_path, file_name)
                shutil.move(src, dest)

print("Files organized successfully!")
