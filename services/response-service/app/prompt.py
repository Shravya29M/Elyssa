import re

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""


def create_prompt(
    user_text: str,
    facial_emotion: str,
    is_conflicting: bool = False,
    text_emotion: str | None = None,
) -> str:
    if is_conflicting:
        instruction = (
            f"Detected facial emotion: {facial_emotion}, but text sentiment suggests: {text_emotion}. "
            f"There seems to be a conflict between facial expression and text content.\n"
            f"You are a helpful mental health counselling assistant. You've noticed a conflict between "
            f"the user's facial expression and their text content. Your first priority is to kindly "
            f"acknowledge this conflict and ask them to confirm their true emotional state.\n"
            f'Begin your response with: "I notice that your facial expression appears {facial_emotion}, '
            f"but your message suggests you might be feeling {text_emotion}. Could you tell me which "
            f'better reflects how you\'re actually feeling right now?"\n'
            f"After that introduction, briefly address their message with empathy, but keep the focus "
            f"on clarifying their emotional state before proceeding with more detailed support."
        )
    else:
        instruction = (
            f"Detected facial emotion: {facial_emotion}.\n\n"
            f"You are a helpful mental health counselling assistant that also considers the detected "
            f"facial emotion of the user, please answer the mental health questions based on the "
            f"patient's description. The assistant gives helpful, comprehensive, and appropriate "
            f"answers to the user's questions."
        )

    return ALPACA_TEMPLATE.format(instruction, user_text, "")


def parse_response(full_decoded: str) -> str:
    match = re.search(r"### Response:\s*(.*)", full_decoded, re.DOTALL)
    if match:
        text = match.group(1).strip()
        return re.sub(r"<\|end_of_text\|>$", "", text).strip()
    return "I apologize, but I couldn't generate a proper response. Please try again."
