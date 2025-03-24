import gradio as gr
import torch
from PIL import Image
import re
import time
import numpy as np
from unsloth import FastLanguageModel, FastVisionModel
from transformers import TextStreamer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK data for sentiment analysis (only needs to be done once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load the Vision model for emotion detection
def load_vision_model():
    print("Loading emotion detection model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "seasalt29/imageModelBig",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer

# Load the Language model for counseling
def load_language_model():
    print("Loading counseling model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="seasalt29/model3",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=False,
    )
    model = FastLanguageModel.for_inference(model)
    return model, tokenizer

# Global variables for models
vision_model, vision_tokenizer = load_vision_model()
language_model, language_tokenizer = load_language_model()
print("Both models loaded successfully!")

# Global variable for storing the last frame
last_frame = None

# Function to store the current webcam frame
def update_frame(frame):
    global last_frame
    if frame is not None:
        last_frame = frame.copy()
    return None

# Function to detect emotion from image - FIXED VERSION
def detect_emotion(image):
    if image is None:
        print("No image provided for emotion detection, returning neutral")
        return "neutral"
        
    # More specific instruction to encourage single-word response
    instruction = "Analyze this facial expression and identify the emotion. Respond with EXACTLY ONE WORD from this list: happy, sad, angry, surprised, fearful, disgusted, neutral, confused, excited, anxious, tired. Your entire response should be just that one word."
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = vision_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    # Convert image to PIL if it's not already
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Process with vision model
    inputs = vision_tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    text_streamer = TextStreamer(vision_tokenizer, skip_prompt=True)
    
    # Lower temperature for more focused responses
    output = vision_model.generate(
        **inputs, 
        streamer=text_streamer, 
        max_new_tokens=128,
        use_cache=True, 
        temperature=0.1,  # Reduced temperature for more decisive output
        min_p=0.05
    )
    
    # Decode the output and extract the emotion word
    decoded = vision_tokenizer.decode(output[0])
    cleaned_text = re.sub(r'<\|.*?\|>', '', decoded).strip().lower()
    
    print(f"Raw model response: '{cleaned_text}'")
    
    # Extract just the emotion word - improved detection
    emotion_words = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", 
                     "neutral", "confused", "excited", "anxious", "tired"]
    
    # FIXED: Improved emotion detection by focusing on the assistant's response
    # First, try to extract just the assistant's part of the response
    assistant_response = ""
    if "assistant" in cleaned_text:
        parts = cleaned_text.split("assistant")
        if len(parts) > 1:
            assistant_response = parts[1].strip()
    
    # If we have an assistant response, use that, otherwise use the full text
    text_to_search = assistant_response if assistant_response else cleaned_text
    
    # Look for exact emotion words with word boundaries
    found_emotion = None
    for word in emotion_words:
        # Use regex with word boundaries to find exact matches
        if re.search(r'\b' + word + r'\b', text_to_search):
            found_emotion = word
            print(f"Found emotion word: {word}")
            break
    
    # If no emotion word is found, try to determine the most similar one
    if found_emotion is None:
        # Default to neutral if no match
        found_emotion = "neutral"
        print(f"No exact emotion match found, defaulting to: {found_emotion}")
    
    print(f"Final detected emotion: {found_emotion}")
    return found_emotion

# Function to analyze text sentiment and categorize emotion
def analyze_text_sentiment(text):
    sentiment = sia.polarity_scores(text)
    
    # Convert sentiment score to emotion category
    if sentiment['compound'] >= 0.5:
        return "happy"  # Strong positive
    elif sentiment['compound'] >= 0.1:
        return "excited"  # Mild positive
    elif sentiment['compound'] <= -0.5:
        return "sad"  # Strong negative
    elif sentiment['compound'] <= -0.1:
        return "anxious"  # Mild negative
    else:
        return "neutral"  # Neutral
        
# Function to detect conflicting emotions
def detect_conflicting_emotions(facial_emotion, text):
    # Get text sentiment
    text_emotion = analyze_text_sentiment(text)
    print(f"Text sentiment detected as: {text_emotion}")
    
    # Define positive, negative, and neutral emotions
    positive_emotions = ["happy", "excited"]
    negative_emotions = ["sad", "angry", "fearful", "disgusted", "anxious", "tired"]
    neutral_emotions = ["neutral", "surprised", "confused"]
    
    # Check for conflicts
    facial_category = ""
    text_category = ""
    
    if facial_emotion in positive_emotions:
        facial_category = "positive"
    elif facial_emotion in negative_emotions:
        facial_category = "negative"
    else:
        facial_category = "neutral"
        
    if text_emotion in positive_emotions:
        text_category = "positive"
    elif text_emotion in negative_emotions:
        text_category = "negative"
    else:
        text_category = "neutral"
    
    # Check if there's a significant conflict
    is_conflicting = (facial_category == "positive" and text_category == "negative") or \
                     (facial_category == "negative" and text_category == "positive")
                     
    return is_conflicting, facial_emotion, text_emotion





# Function to create the prompt
def create_prompt(user_text, detected_emotion, conflicting=False, text_emotion=None):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""
    if conflicting:
        # Add this print statement to output HELLOOOO when conflicting emotions are detected

        instruction = f"""Detected facial emotion: {detected_emotion}, but text sentiment suggests: {text_emotion}. There seems to be a conflict between facial expression and text content.
You are a helpful mental health counselling assistant. You've noticed a conflict between the user's facial expression and their text content. Your first priority is to kindly acknowledge this conflict and ask them to confirm their true emotional state.
Begin your response with: "I notice that your facial expression appears {detected_emotion}, but your message suggests you might be feeling {text_emotion}. Could you tell me which better reflects how you're actually feeling right now?"
After that introduction, briefly address their message with empathy, but keep the focus on clarifying their emotional state before proceeding with more detailed support."""
    else:
        instruction = f"Detected facial emotion: {detected_emotion}\n\n. You are a helpful mental health counselling assistant that also considers the detected facial emotion of the user, please answer the mental health questions based on the patient's description. The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."

    return alpaca_prompt.format(instruction, user_text, "")










    
# Function to generate counseling response
def generate_response(user_text, user_image):
    print("\n--- Starting response generation ---")
    print(f"User message: '{user_text}'")
    
    # First, detect the emotion from the image
    facial_emotion = detect_emotion(user_image)
    
    # Check for conflicting emotions
    is_conflicting, facial_emotion, text_emotion = detect_conflicting_emotions(facial_emotion, user_text)
    print(f"Facial emotion: {facial_emotion}, Text emotion: {text_emotion}")
    print(f"Emotions conflicting: {is_conflicting}")
    
    # Create the prompt with the detected emotion and conflict status
    prompt = create_prompt(user_text, facial_emotion, is_conflicting, text_emotion)
    
    # Generate response using the language model
    inputs = language_tokenizer(
        [prompt], 
        return_tensors="pt"
    ).to("cuda")
    
    output = language_model.generate(
        **inputs, 
        max_new_tokens=512, 
        use_cache=True
    )
    
    # Decode the response
    full_response = language_tokenizer.decode(output[0])
    
    # Extract just the response part and clean up any special tokens
    response_match = re.search(r'### Response:\s*(.*)', full_response, re.DOTALL)
    if response_match:
        response = response_match.group(1).strip()
        # Remove any end of text tokens that might appear in the response
        response = re.sub(r'<\|end_of_text\|>$', '', response)
    else:
        response = "I apologize, but I couldn't generate a proper response. Please try again."
    
    emotion_to_display = facial_emotion
    if is_conflicting:
        print("I notice there seems to be a difference between your facial expression and what you've shared in your message. This happens naturally sometimes. If my response doesn't quite address what you're feeling, would you feel comfortable sharing which emotion is closer to your experience right now? Understanding your true feelings will help me support you better.")
        emotion_to_display = f"{facial_emotion} (face) vs {text_emotion} (text) - conflicting. \n\n I notice there seems to be a difference between your facial expression and what you've shared in your message. This happens naturally sometimes. If my response doesn't quite address what you're feeling, would you feel comfortable sharing which emotion is closer to your experience right now? Understanding your true feelings will help me support you better."
    
    print(f"Response generated with emotion: {emotion_to_display}")
    print("--- Response generation complete ---\n")
    
    return response, emotion_to_display











# Function to add visual feedback to the captured image
def process_image(img):
    if img is None:
        return None
    
    # Add a visual indicator that the photo was taken
    processed = img.copy()
    if processed is not None:
        # Add a white frame to indicate capture
        h, w = processed.shape[0], processed.shape[1]
        frame_thickness = 10
        processed[0:frame_thickness, :] = [255, 255, 255]  # Top border
        processed[h-frame_thickness:h, :] = [255, 255, 255]  # Bottom border
        processed[:, 0:frame_thickness] = [255, 255, 255]  # Left border
        processed[:, w-frame_thickness:w] = [255, 255, 255]  # Right border
        
    return processed

# Chat function with auto-capture
def add_text(history, text):
    global last_frame
    
    if not text.strip():
        return history, "", None, "Whenever you're ready, just type a message—we're here to listen."
    
    # Use the last frame that was captured
    if last_frame is not None:
        # Process the image for display
        processed_image = process_image(last_frame)
    else:
        processed_image = None
        return history, text, None, "No webcam image detected. Please make sure your camera is working."
    
    # Add message to history
    history = history + [(text, None)]
    return history, "", processed_image, "Photo automatically captured for emotion analysis"

def bot(history):
    global last_frame
    
    # Get the last user message
    user_message = history[-1][0]
    
    # Generate response with emotion detection
    response, emotion = generate_response(user_message, last_frame)
    
    # Update history with bot response and emotion info
    history[-1] = (user_message, f"{response}\n\n[Detected emotion: {emotion}]")
    
    return history

# Main Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Elyssa – Bringing Therapy Closer to You
        
        Elyssa sees beyond your words. As you share your thoughts, our caring system quietly observes the emotions written on your face – those subtle signals we often don't express in text. Just be yourself as you type, and Elyssa will blend what you say with what your expressions reveal, creating a space where support meets you exactly where you are. Your complete emotional story helps us journey with you more meaningfully
        """
    )
    
    # Status message (for feedback)
    status = gr.Textbox(
        value="Your webcam is streaming! No rush, take a deep breath, and begin when you’re ready-—we’re always here for you.",
        label="Status",
        interactive=False
    )
    
    # Two-column layout
    with gr.Row():
        # Left column: webcam and captured image
        with gr.Column(scale=1):
            # In Gradio 5.x with automatic capture, we need to use a streaming webcam
            webcam = gr.Image(
                label="Live Camera Feed",
                type="numpy",
                sources=["webcam"],
                streaming=True,
                height=200
            )
            
            # Display of captured image with visual feedback
            captured = gr.Image(
                label="Last Captured Image",
                type="numpy",
                height=200
            )
            
        # Right column: chat interface
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                bubble_full_width=False,
                show_copy_button=True,
                avatar_images=("👤", "🤖"),
            )
            
            # Text input and buttons
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here and press Enter or click Send...",
                    container=False,
                    scale=6,
                    show_label=False,
                )
                send_btn = gr.Button("Send", scale=1)
                clear = gr.Button("Clear Chat", scale=1)
    
    # Set up webcam continuous frame capture
    webcam.stream(
        update_frame,
        inputs=[webcam],
        outputs=None,
        show_progress=False
    )
    
    # Set up event handlers
    # Handle message submission (both Enter key and Send button)
    msg.submit(
        add_text,
        [chatbot, msg],
        [chatbot, msg, captured, status]
    ).then(
        bot,
        [chatbot],
        chatbot
    )
    
    # Send button does the same as Enter key
    send_btn.click(
        add_text,
        [chatbot, msg],
        [chatbot, msg, captured, status]
    ).then(
        bot,
        [chatbot],
        chatbot
    )
    
    # Clear button
    clear.click(lambda: [], outputs=[chatbot])
    
    # Example prompts
    gr.Examples(
        [
            "I passed all my exams right now. I feel so happy and content.",
            "I'm feeling overwhelmed with my workload lately.",
            "I had a fight with my friend and I'm not sure what to do.",
            "I've been feeling down for the past few weeks."
        ],
        inputs=msg
    )
    
    # Add debug button
    with gr.Row():
        debug_btn = gr.Button("Debug Emotion Detection")
        debug_output = gr.Textbox(label="Debug Information", lines=5)
        
    def run_debug():
        global last_frame
        if last_frame is not None:
            facial_emotion = detect_emotion(last_frame)
            # For testing conflict detection, just use a dummy text
            test_text = "This is a test message to debug the emotion detection."
            is_conflicting, facial_emotion, text_emotion = detect_conflicting_emotions(facial_emotion, test_text)
            return f"Current facial emotion: {facial_emotion}\nText sentiment: {text_emotion}\nConflicting emotions: {is_conflicting}\nCheck terminal/console for more detailed debug info."
        else:
            return "No image available for testing. Please make sure your webcam is active."
            
    debug_btn.click(run_debug, inputs=[], outputs=debug_output)
    
    # Privacy notice
    gr.Markdown(
        """
        ### Privacy Notice
        
        This application automatically captures your webcam images when you send a message to detect emotions.
        Images are processed but not stored. By using this application, you consent to facial
        emotion analysis for mental health support purposes.
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.queue().launch()