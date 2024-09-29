import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info  # Ensure this function is correctly imported

# Load the Qwen2-VL model and processor
@st.cache_resource
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")  # Removed torch_dtype and .cuda()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    return model, processor

model, processor = load_model()

# Function to extract text from image
def extract_text_from_image(image):
    image = image.convert("RGB")  # Convert image to RGB

    # Prepare the message structure
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": ""}]}]

    # Preprocess the inputs
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[""], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")  # Optional: You can comment this line out if not using GPU

    # Generate the output text
    generated_ids = model.generate(**inputs, max_new_tokens=256)  # Increased tokens to capture more text
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output_text[0]

# Streamlit application
st.title("OCR with Qwen2-VL")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract text using the model
    st.write("Extracting text...")
    extracted_text = extract_text_from_image(image)
    st.write("Extracted Text:")
    st.write(extracted_text)

    # Search functionality
    search_query = st.text_input("Search within extracted text")
    if search_query:
        if search_query.lower() in extracted_text.lower():
            st.write(f"'{search_query}' found in the extracted text.")
        else:
            st.write(f"'{search_query}' not found in the extracted text.")
