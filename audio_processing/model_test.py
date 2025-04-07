import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
# from pydub import AudioSegment
from PIL import Image
import os
import re
import io
import itertools
import pyaudio
import wave
import time
import matplotlib
import threading

import torch

from torchvision.transforms import Compose, ToTensor, Grayscale, Resize, Normalize

from torchvision.transforms import (
    Grayscale, ToTensor, Compose, Resize, InterpolationMode, Normalize, Lambda
)
import torch.nn.functional as F




# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono recording
RATE = 16000  # Sampling rate in Hz
CHUNK = 1024  # Buffer size
RECORD_SECONDS = 3


def record_audio_to_numpy():
    audio = pyaudio.PyAudio()
    
    # Open the microphone stream
    
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    # print("Recording...")
    frames = []
    
    # Read data from the stream
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    # print("Recording finished.")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Convert recorded frames to numpy array
    audio_array = np.frombuffer(b''.join(frames), dtype=np.int16)
    return audio_array

def numpy_to_fft(samples, sr = 16000):
    n_fft=2048 
    win_length1 = 750 
    hop_length=win_length1//4 

    # Convert to NumPy array and normalize
    samples = samples.astype(np.float32)/float(np.max(samples))

    # Compute spectrogram using STFT
    S = librosa.stft(samples, n_fft=n_fft,win_length = win_length1, hop_length=hop_length)  

    #The number of rows in the STFT matrix D is (1 + n_fft/2).
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    S0 = S_db

    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)  # 256x256 pixels
    librosa.display.specshow(S_db, sr=sr, n_fft=n_fft, win_length=win_length1, hop_length=hop_length, x_axis="time", y_axis="log", cmap="gray")

    # Remove axes for a clean image
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    # print(plt.ylim())
    plt.ylim([30,6000])
    buf = io.BytesIO()
    # plt.clf()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)  # Move cursor to the start of the buffer

    # Process the image from memory
    img = Image.open(buf).convert("L")  # Convert to grayscale
    img = img.resize((256, 256))  # Resize to 256x256

    # Convert to NumPy array
    # img_array = np.array(img, dtype=np.uint8)
    # img_array = img_array[np.newaxis, :, :]  # Add channel dimension (1,256,256)
    
    # Close buffer
    buf.close()
    return img#Image.fromarray(img_array.squeeze())


def fft_to_tensor(pil_image):
   
    image = pil_image

    # Rotate 90 degrees (optional, remove if unnecessary)
    image = image.transpose(Image.ROTATE_90)

    # Ensure the image is in RGB mode (some formats might be grayscale)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    transform = Compose([
                            ToTensor(),
                            Grayscale(),
                            Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                            Normalize(mean=[0.5], std=[0.5])
                        ])
    

    input_tensor = transform(image)  # Shape: (1, 224, 224)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension -> (1, 1, 224, 224)
    if input_tensor.shape[1] == 1:  # Convert grayscale to RGB
        input_tensor = input_tensor.expand(-1, 3, -1, -1)  # Shape: (1, 3, H, W)

    # Apply transformations and return image with label
    return input_tensor

def evaluate(output, threshold = .9):
    # Apply softmax to convert logits to probabilities
    probabilities = np.array(F.softmax(output, dim=1).squeeze())  # Shape: (num_classes,)
    # Class labels
    guess = np.argmax(probabilities)
    class_labels = ['arabic', 'english', 'german', 'mandarin', 'spanish', 'garbage', 'french']

    # Display probabilities
    language = class_labels[guess]
    confidence = probabilities[guess]

    # if np.random.rand()<.2: language = "english!"
    if confidence>threshold:
        return language, confidence
    
    return f"low confidence({language})", confidence


# def main():
#     os.chdir(r"C:\Git_repos\ENDG 511\ENDG511_Final_Project\audio_processing")
#     image = None
#     model_path = r"C:\Git_repos\ENDG 511\ENDG511_Final_Project\models\model_language_fix_data_10_epoch.pth"
#     model = torch.load(model_path, map_location="cpu", weights_only=False)
#     model.eval()  # Set to evaluation mode

#     # set up the plot window
#     matplotlib.use('TkAgg')  # Use a GUI backend for external window
#     # plt.ion()
#     # fig, ax = plt.subplots()
#     # img1 = Image.open("spectrogram.png")
#     # im_display = ax.imshow(img1)
#     # title = ax.set_title("Image 1")  # Set initial title
#     # plt.axis('off')  # Hide axes
#     # plt.show()

#     plt.ion()
#     fig, ax = plt.subplots()
#     img1 = Image.open("spectrogram.png")
#     im_display = ax.imshow(img1, animated=True)
#     title = ax.set_title("Image 1")
#     plt.axis('off')
#     plt.show(block=False)

#     for i in range(15):
#         # generate image and predict with model
#         array = record_audio_to_numpy()
#         image = numpy_to_fft(array) #generate a pil image of the spectrogram
#         input_tensor = fft_to_tensor(image)
#         with torch.no_grad(): output = model(input_tensor) 
#         language, confidence = evaluate(output)
#         text_out = language, f'{round(confidence*100, 3)}%'
#         print(text_out)

#         #update the plot window
#         image.save("spectrogram.png")
#         img2 = Image.open("spectrogram.png")
#         im_display.set_data(img2)
#         title.set_text(text_out)  # Update title

#         # fig.canvas.draw()
#         # fig.canvas.flush_events()
#         fig.canvas.draw_idle()  # More responsive than draw()
#         plt.pause(0.1)  # Allow GUI to update smoothly
def main():
    import os
    os.chdir(r"C:\Git_repos\ENDG 511\ENDG511_Final_Project\audio_processing")

    model_path = r"C:\Git_repos\ENDG 511\ENDG511_Final_Project\models\model_language_fix_data_10_epoch.pth"
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()  # Set to evaluation mode


    # GUI setup in main thread
    plt.ion()
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
    im_display = ax.imshow(np.zeros((256, 256)), cmap='gray')
    title = ax.set_title("Listening...")
    plt.axis('off')
    plt.show(block=False)

    stop_flag = threading.Event()

    def background_loop():
        for _ in range(15):
            if stop_flag.is_set(): break

            array = record_audio_to_numpy()
            image = numpy_to_fft(array)
            tensor = fft_to_tensor(image)

            with torch.no_grad():
                output = model(tensor)

            language, conf = evaluate(output)
            text_out = f"{language} - {round(conf * 100, 3)}%"
            print(text_out)

            img_arr = np.asarray(image)

            def update_plot():
                im_display.set_data(img_arr)
                title.set_text(text_out)
                fig.canvas.draw_idle()

            # Schedule GUI update from thread
            try:
                fig.canvas.manager.window.after(0, update_plot)
            except AttributeError:
                # Backend not ready (usually harmless if window is closing)
                break

            time.sleep(0.5)

    # Start background thread
    thread = threading.Thread(target=background_loop, daemon=True)
    thread.start()

    try:
        while thread.is_alive() and plt.fignum_exists(fig.number):
            plt.pause(0.05)
    except KeyboardInterrupt:
        stop_flag.set()
        thread.join()

if __name__ == "__main__": main()