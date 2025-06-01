# app.py - Final Fixed Version with Correct Prediction Interpretation
import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import tensorflow as tf

# ===== Configuration =====
TFLITE_MODEL_PATH = "model_unquant.tflite"
CLASSES = ["Good", "Defective"]  # Must match Teachable Machine's order

# ===== Model Loader =====
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# ===== Correct Prediction Logic =====
def get_prediction(interpreter, img_array):
    # Get model input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    # Get output - critical fix here
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Teachable Machine binary models typically output:
    # [[good_prob, defective_prob]] for each image
    if output_data.shape[-1] == 2:  # Two output classes
        good_prob = output_data[0][0]
        defect_prob = output_data[0][1]
        class_id = np.argmax(output_data[0])
        confidence = max(good_prob, defect_prob)
    else:  # Fallback for single output
        class_id = 1 if output_data[0][0] > 0.5 else 0
        confidence = output_data[0][0] if class_id == 1 else 1 - output_data[0][0]
    
    return CLASSES[class_id], float(confidence)

# ===== Image Processing =====
def preprocess(img, size=(224, 224)):
    img = cv2.resize(img, size)
    img = (img / 127.5) - 1.0  # Teachable Machine's normalization
    return np.expand_dims(img, axis=0).astype(np.float32)

# ===== Video Processor =====
class FinalVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.interpreter = load_model()
        self.threshold = 0.5  # Adjust if needed
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_input = preprocess(img)
        
        label, confidence = get_prediction(self.interpreter, img_input)
        
        # Visual feedback
        color = (0, 255, 0) if label == "Good" else (0, 0, 255)
        text = f"{label} ({confidence:.1%})"
        cv2.putText(img, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if label == "Defective":
            img = cv2.rectangle(img, (0, 0), 
                              (img.shape[1], img.shape[0]), 
                              (0, 0, 255), 5)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===== Streamlit UI =====
st.title("🔍 Final Capsule Inspector")
st.markdown("""
**Correct prediction interpretation**  
Matches Teachable Machine's output format exactly
""")

# Add confidence threshold slider
threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05,
                     help="Adjust sensitivity for defect detection")

mode = st.radio("Input Mode:", ("Live Camera", "Image Upload"), horizontal=True)

if mode == "Live Camera":
    ctx = webrtc_streamer(
        key="final-inspector",
        video_processor_factory=FinalVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
else:
    uploaded_file = st.file_uploader("Upload capsule image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        interpreter = load_model()
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)
        
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        label, confidence = get_prediction(interpreter, preprocess(img_array))
        
        # Apply threshold
        final_label = label if confidence >= threshold else "Uncertain"
        
        st.subheader("Results")
        if final_label == "Defective":
            st.error(f"❌ Defective (Confidence: {confidence:.1%})")
        elif final_label == "Good":
            st.success(f"✅ Good (Confidence: {confidence:.1%})")
        else:
            st.warning(f"⚠️ Uncertain (Confidence: {confidence:.1%} < {threshold:.0%})")

# ===== Model Inspection =====
with st.expander("Model Debug Info"):
    if st.button("Inspect Model Output"):
        try:
            interpreter = load_model()
            output_details = interpreter.get_output_details()
            input_details = interpreter.get_input_details()
            
            st.write("**Input Details:**")
            st.json(input_details[0])
            
            st.write("**Output Details:**")
            st.json(output_details[0])
            
            # Test with dummy data
            dummy_input = np.ones(input_details[0]['shape'], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            dummy_output = interpreter.get_tensor(output_details[0]['index'])
            
            st.write("**Dummy Test Output:**")
            st.write(dummy_output)
            st.write(f"Output Shape: {dummy_output.shape}")
            
        except Exception as e:
            st.error(f"Error inspecting model: {str(e)}")