import os
import random
import shutil

# 1. Get a list of all .jpg files in the folder
all_images = [f for f in os.listdir('.') if f.endswith('.jpg')]

# 2. Shuffle to make the selection random
random.shuffle(all_images)

# 3. Create 4 folders and distribute 42 images to each
for i in range(4):
    folder_name = f"subset_{i+1}"
    os.makedirs(folder_name, exist_ok=True)
    
    # Get 42 images for this specific subset
    subset = all_images[i*42 : (i+1)*42]
    
    # Copy images into the new folder
    for img in subset:
        shutil.copy(img, os.path.join(folder_name, img))
    
    print(f"Successfully created {folder_name} with 42 images.")

print("Done! You now have 4 subset folders with 42 images each.")