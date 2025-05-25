"""
End-of-turn detection Python implementation
Original source: LiveKit Agents Project
License: Apache License 2.0
"""

from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time

# Constants
HG_MODEL = "livekit/turn-detector"
ONNX_FILENAME = "model_q8.onnx"
MODEL_REVISION = "v0.2.0-intl"
MAX_HISTORY_TURNS = 6
MAX_HISTORY_TOKENS = 128
MIN_LANGUAGE_DETECTION_LENGTH = 5

UNLIKELY_THRESHOLD = 0.1

chat_example1 = [
    {"role": "user", "content": "今天天气怎么样？"},
    {"role": "assistant", "content": "今天阳光明媚，温度适中。"},
    #{"role": "user", "content": "我很喜欢这样的天气，不过"},
    {"role": "user", "content": "我不太确定要不要, 出去玩吧"}
]

chat_example2 = [
    {"role": "user", "content": "你能帮我查下附近的餐厅吗？"},
    {"role": "assistant", "content": "好的，我可以帮你找到周边的美食。"},
    {"role": "user", "content": "我想吃中餐"},
    {"role": "assistant", "content": "我推荐你去尝试一下那家新开的川菜馆。"},
]

chat_example3 = [
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "It's sunny and warm."},
    {"role": "user", "content": "I like the weather. but"},
    {"role": "user", "content": "I'm not sure what to do?"}
]

def initialize_model():
    """
    Initialize the ONNX model and tokenizer.

    Returns:
        tuple: (tokenizer, session)
    """
    try:
        start_time = time.time()

        # Get the current file's directory and construct model path
        model_path = f'models1/{ONNX_FILENAME}'
        session = ort.InferenceSession(str(model_path))

        tokenizer = AutoTokenizer.from_pretrained('models1', truncation_side="left")

        print(f"Model initialization took: {time.time() - start_time:.2f} seconds")
        return tokenizer, session

    except Exception as e:
        print(f"Error: {e}")
        raise

def normalize(text):
    """
    Normalize the input text by removing punctuation and standardizing whitespace.
    去除标点符号并将所有文本转换为小写，确保模型关注内容而不是格式变体。
    """
    PUNCS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'  # Punctuation without single quote

    # Remove punctuation and normalize whitespace
    stripped = ''.join(char for char in text if char not in PUNCS)
    return ' '.join(stripped.lower().split())

def format_chat_ctx(chat_ctx, tokenizer):
    """Format the chat context for model input."""
    new_chat_ctx = []
    for msg in chat_ctx:
        content = msg["content"]
        if not content:
            continue

        msg["content"] = content
        new_chat_ctx.append(msg)

    convo_text = tokenizer.apply_chat_template(
        new_chat_ctx,
        add_generation_prompt=False,
        add_special_tokens=False,
        tokenize=False,
    )

    # remove the EOU token from current utterance
    ix = convo_text.rfind("<|im_end|>")
    text = convo_text[:ix]
    return text

def softmax(logits):
    """
    Compute softmax probabilities for logits.
    """
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def predict_end_of_turn(chat_context, model_data):
    """
    Predict whether the current turn is complete.

    Args:
        chat_context (list): List of chat messages
        model_data (tuple): (tokenizer, session)

    Returns:
        float: Probability of end of turn
    """
    tokenizer, session = model_data

    formatted_text = format_chat_ctx(chat_context, tokenizer)

    inputs = tokenizer(
            formatted_text,
            add_special_tokens=False,
            return_tensors="np",  # ONNX requires NumPy format
            max_length=MAX_HISTORY_TOKENS,
            truncation=True,
        )

    outputs = session.run(None, {"input_ids": inputs["input_ids"].astype("int64")})
    #print(f"Model outputs: {outputs}")
    eou_probability = outputs[0].flatten()[-1]
    return eou_probability

def main():
    """
    Main function to demonstrate usage.
    """
    model_data = initialize_model()

    start_time = time.time()

    # Run predictions
    for i, example in enumerate([chat_example1[-MAX_HISTORY_TURNS:], chat_example2[-MAX_HISTORY_TURNS:], chat_example3[-MAX_HISTORY_TURNS:]], 1):
        probability = predict_end_of_turn(example, model_data)
        print(f'End of turn probability{i}: {probability}')

    print(f"If probability is less than {UNLIKELY_THRESHOLD}, "
          "the model predicts that the user hasn't finished speaking.")

    print(f"Prediction time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
