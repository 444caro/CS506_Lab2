import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# Function to load and preprocess the image
def load_image(image_path):
    #Open the image file using PIL
    image = Image.open(image_path)
    #Convert the image to NumPy array
    image_np = np.array(image)
    return image_np
    #raise NotImplementedError('You need to implement this function')

# Function to perform KMeans clustering for image quantization
def image_compression(image_np, n_colors):
    # reshape image to 2D array of pixels
    width, height, depth = image_np.shape
    image_reshaped = image_np.reshape(image_np, (width * height, depth))
    
    # apply KMeans clustering to the reshaped image
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(image_reshaped)
    #replace pixel values with their corresponding cluster centers
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
    #reshape the compressed pixels back to the original image shape
    compressed_image_np = compressed_pixels.reshape((width, height, depth)).astype(np.uint8)
    return compressed_image_np
    
    #raise NotImplementedError('You need to implement this function')

# Function to concatenate and save the original and quantized images side by side
def save_result(original_image_np, quantized_image_np, output_path):
    # Convert NumPy arrays back to PIL images
    original_image = Image.fromarray(original_image_np)
    quantized_image = Image.fromarray(quantized_image_np)
    
    # Get dimensions
    width, height = original_image.size
    
    # Create a new image that will hold both the original and quantized images side by side
    combined_image = Image.new('RGB', (width * 2, height))
    
    # Paste original and quantized images side by side
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(quantized_image, (width, 0))
    
    # Save the combined image
    combined_image.save(output_path)

def __main__():
    # Load and process the image
    image_path = 'favorite_image.png'  
    output_path = 'compressed_image.png'  
    image_np = load_image(image_path)

    # Perform image quantization using KMeans
    n_colors = 8  # Number of colors to reduce the image to, you may change this to experiment
    quantized_image_np = image_compression(image_np, n_colors)

    # Save the original and quantized images side by side
    save_result(image_np, quantized_image_np, output_path)
