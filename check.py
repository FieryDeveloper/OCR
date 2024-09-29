
import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info

@st.cache_resource
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    return model, processor

model, processor = load_model()
text_query = "Extract only the text from the image. Do not include any other information."

def extract_text_from_image(image, text_query):
    image = image.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image  
                },
                {"type": "text", "text": text_query},  # The text query
            ],
        }
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Removed GPU functionality
    inputs = inputs.to("cpu")  # Ensure inputs are on CPU

    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return output_text[0]

st.title("OCR with Qwen2-VL")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Extracting text...")
    extracted_text = extract_text_from_image(image, text_query)
    st.write("Extracted Text:")
    st.write(extracted_text)

    search_query = st.text_input("Search within extracted text")
    if search_query:
        highlighted_text = extracted_text.replace(search_query, f"<mark>{search_query}</mark>")
        st.markdown("Extracted Text with Highlighted Keyword:")
        st.markdown(highlighted_text, unsafe_allow_html=True)
