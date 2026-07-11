import base64
import io
import os

import gradio as gr
import httpx
from PIL import Image

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")


def update_frame(frame, last_frame):
    """Keep the latest webcam frame in per-session state."""
    if frame is not None:
        return frame.copy()
    return last_frame


def _frame_to_b64(frame) -> str | None:
    """Convert a numpy frame to a base64-encoded JPEG string."""
    if frame is None:
        return None
    pil_img = Image.fromarray(frame)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def process_image(img):
    """Add a white border to indicate image capture."""
    if img is None:
        return None
    processed = img.copy()
    h, w = processed.shape[0], processed.shape[1]
    t = 10
    processed[0:t, :] = [255, 255, 255]
    processed[h - t : h, :] = [255, 255, 255]
    processed[:, 0:t] = [255, 255, 255]
    processed[:, w - t : w] = [255, 255, 255]
    return processed


def add_text(history, text, last_frame):
    if not text.strip():
        return history, "", None, "Whenever you're ready, just type a message — we're here to listen."
    if last_frame is None:
        return history, text, None, "No webcam image detected. Please make sure your camera is working."
    processed_image = process_image(last_frame)
    history = history + [(text, None)]
    return history, "", processed_image, "Photo automatically captured for emotion analysis"


def bot(history, last_frame):
    user_message = history[-1][0]
    image_b64 = _frame_to_b64(last_frame)

    if image_b64 is None:
        history[-1] = (user_message, "Could not capture webcam image. Please ensure your camera is active.")
        return history

    try:
        resp = httpx.post(
            f"{GATEWAY_URL}/chat",
            json={"user_text": user_message, "image_b64": image_b64},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        response_text = data.get("response_text", "")
        emotion_display = data.get("emotion_display", "unknown")
        history[-1] = (user_message, f"{response_text}\n\n[Detected emotion: {emotion_display}]")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            history[-1] = (
                user_message,
                "The response service is temporarily unavailable. Please try again in a moment.",
            )
        else:
            history[-1] = (user_message, f"An error occurred (HTTP {e.response.status_code}). Please try again.")
    except Exception as e:
        history[-1] = (user_message, f"Could not reach the assistant: {e}")

    return history


# Build Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Elyssa – Bringing Therapy Closer to You

        Elyssa sees beyond your words. As you share your thoughts, our caring system quietly observes
        the emotions written on your face – those subtle signals we often don't express in text.
        Just be yourself as you type, and Elyssa will blend what you say with what your expressions
        reveal, creating a space where support meets you exactly where you are.
        """
    )

    status = gr.Textbox(
        value="Your webcam is streaming! No rush, take a deep breath, and begin when you're ready — we're always here for you.",
        label="Status",
        interactive=False,
    )

    with gr.Row():
        with gr.Column(scale=1):
            webcam = gr.Image(
                label="Live Camera Feed",
                type="numpy",
                sources=["webcam"],
                streaming=True,
                height=200,
            )
            captured = gr.Image(
                label="Last Captured Image",
                type="numpy",
                height=200,
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                bubble_full_width=False,
                show_copy_button=True,
                avatar_images=("👤", "🤖"),
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here and press Enter or click Send...",
                    container=False,
                    scale=6,
                    show_label=False,
                )
                send_btn = gr.Button("Send", scale=1)
                clear = gr.Button("Clear Chat", scale=1)

    # Per-session frame storage: prevents one user's webcam frame from
    # leaking into another user's request in multi-user deployments.
    frame_state = gr.State(None)

    webcam.stream(
        update_frame, inputs=[webcam, frame_state], outputs=frame_state, show_progress=False
    )

    msg.submit(add_text, [chatbot, msg, frame_state], [chatbot, msg, captured, status]).then(
        bot, [chatbot, frame_state], chatbot
    )
    send_btn.click(add_text, [chatbot, msg, frame_state], [chatbot, msg, captured, status]).then(
        bot, [chatbot, frame_state], chatbot
    )
    clear.click(lambda: [], outputs=[chatbot])

    gr.Examples(
        [
            "I passed all my exams right now. I feel so happy and content.",
            "I'm feeling overwhelmed with my workload lately.",
            "I had a fight with my friend and I'm not sure what to do.",
            "I've been feeling down for the past few weeks.",
        ],
        inputs=msg,
    )

    gr.Markdown(
        """
        ### Privacy Notice

        This application automatically captures your webcam images when you send a message to detect emotions.
        Images are processed in real time and are not stored. By using this application, you consent to facial
        emotion analysis for mental health support purposes.
        """
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
