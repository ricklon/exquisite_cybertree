import os
import requests
import time
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import base64
import ffmpeg  # Import ffmpeg-python
import json

# Load the environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the API key for OpenAI
OpenAI.api_key = openai_api_key

# Import OpenAI client
client = OpenAI()

# Generate an image from a caption
def generate_image(caption):
    response = client.images.generate(
        model="dall-e-3",
        prompt=caption,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    url = response.data[0].url

    image = requests.get(url).content
    image = plt.imread(BytesIO(image))

    return image

# Find the caption from the image
def generate_caption(image):
    image_data = BytesIO()
    plt.imsave(image_data, image, format='png')
    image_data.seek(0)

    # Encode the byte stream to base64
    image_base64 = base64.b64encode(image_data.read()).decode('utf-8')
   
    # Prepare the payload
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What might the prompt be for this image? Provide a creative and slightly altered description."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 300
    }

    # Make the request to OpenAI
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {client.api_key}"
        },
        json=payload
    )
    
    # Extract the JSON data from the response
    response_data = response.json()
    caption = response_data["choices"][0]["message"]["content"]
    
    # Introduce variability
    words = caption.split()
    np.random.shuffle(words)
    shuffled_caption = ' '.join(words)
    
    return shuffled_caption

# Resize an image using ffmpeg
def resize_image(input_image, output_image, size=(1024, 1024)):
    input_file = BytesIO()
    plt.imsave(input_file, input_image, format='png')
    input_file.seek(0)
    
    temp_input = "temp_input.png"
    temp_output = output_image
    with open(temp_input, 'wb') as f:
        f.write(input_file.read())

    try:
        (
            ffmpeg
            .input(temp_input)
            .filter('scale', size[0], size[1])
            .output(temp_output, vframes=1, format='image2')
            .overwrite_output()
            .run()
        )
        if os.path.exists(temp_output):
            st.write(f"Resized image saved: {temp_output}")
        else:
            st.write(f"Failed to save resized image: {temp_output}")
    except ffmpeg.Error as e:
        st.write(f"ffmpeg error: {e.stderr.decode()}")
    finally:
        os.remove(temp_input)

# Create a video from images using ffmpeg
def make_video(image_files, output_filename, image_duration=2.5):
    try:
        temp_dir = os.path.dirname(image_files[0])
        temp_pattern = os.path.join(temp_dir, "image_%d.png")
        
        # Rename images to follow the pattern
        for i, image_file in enumerate(image_files):
            new_name = temp_pattern % i
            os.rename(image_file, new_name)
        
        (
            ffmpeg
            .input(temp_pattern, framerate=1/image_duration)
            .output(output_filename, vcodec="libx264", format="mp4", pix_fmt="yuv420p")
            .run(overwrite_output=True)
        )
        if os.path.exists(output_filename):
            st.write(f"Video file created: {output_filename}")
        else:
            st.write(f"Failed to create video file: {output_filename}")
    except ffmpeg.Error as e:
        st.write(f"ffmpeg error: {e.stderr.decode()}")
    except Exception as ex:
        st.write(f"Error: {str(ex)}")

# Run the exquisite corpse game
def run_excorp(num_turns=5):
    # Create a directory for this run
    run_id = int(time.time())
    run_dir = f"exquisite_corpse_run_{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    # Create subdirectories for images and videos
    images_dir = os.path.join(run_dir, "images")
    videos_dir = os.path.join(run_dir, "videos")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    # Metadata file
    metadata_file = os.path.join(run_dir, "metadata.json")
    metadata = {"run_id": run_id, "captions": []}

    st.write(f"Welcome to the Exquisite Corpse Game! Total turns: {num_turns}")
    # Initialize the first caption
    caption = "Start of the exquisite cyber tree game"

    # Initialize an empty list for images and captions
    images_list = []
    captions = [caption]

    # Loop for the specified number of turns
    for turn in range(num_turns):
        st.write(f"Turn {turn+1}/{num_turns}")
        # Generate the image
        image = generate_image(caption)
        
        # Save the image with a unique filename
        filename = os.path.join(images_dir, f"image_turn_{turn+1}.png")
        plt.imsave(filename, image)
        st.write(f"Saved image: {filename}")

        # Resize the image
        resized_filename = os.path.join(images_dir, f"resized_image_turn_{turn+1}.png")
        resize_image(image, resized_filename)

        # Add image to list of images
        images_list.append(resized_filename)

        # Display the image in Streamlit
        st.image(image, caption=f"Turn {turn+1}: {caption}", use_column_width=True)

        # Generate the caption
        caption = generate_caption(image)

        # Add caption to list of captions
        captions.append(caption)
        
        # Print the caption
        st.write("Caption:", caption)

    # Save metadata
    metadata["captions"] = captions
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    # Make video from images
    video_filename = os.path.join(videos_dir, f"video_{run_id}.mp4")
    make_video(images_list, video_filename)
    
    # Debugging: Check current directory and list files
    current_dir = os.getcwd()
    st.write(f"Current directory: {current_dir}")
    st.write(f"Files in the current directory: {os.listdir(current_dir)}")
    
    # Verify if video file exists
    if os.path.exists(video_filename):
        st.write(f"Video file {video_filename} exists.")
        st.video(video_filename)
    else:
        st.write(f"Video file {video_filename} does not exist.")

# Streamlit UI
st.title("Exquisite Corpse Game")
num_turns = st.slider("Select number of turns", min_value=1, max_value=10, value=5)
if st.button("Start Game"):
    run_excorp(num_turns)
