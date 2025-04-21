import logging
logger = logging.getLogger(__name__)

import transformers
import collections
import threading
import datetime
import queue
import torch
import time
import re

model_dir_local = "KoljaB/SentenceFinishedClassification"
model_dir_cloud = "/root/models/sentenceclassification/"
sentence_end_marks = ['.', '!', '?', '。']

# fast settings:
# detection_speed = 0.5
# ellipsis_pause = 2.3
# punctuation_pause = 0.39
# exclamation_pause = 0.35
# question_pause = 0.33
# unknown_sentence_detection_pause = 1.25

# slow settings:
detection_speed = 1.0
ellipsis_pause = 2.8
punctuation_pause = 0.56
exclamation_pause = 0.53
question_pause = 0.50
unknown_sentence_detection_pause = 1.7

anchor_points = [
    (0.0, 1.0),
    (1.0, 0.0)
]

def ends_with_string(text: str, s: str):
    if text.endswith(s):
        return True
    if len(text) > 1 and text[:-1].endswith(s):
        return True
    return False

def get_suggested_whisper_pause(text):
    if ends_with_string(text, "..."):
        return ellipsis_pause
    elif ends_with_string(text, "."):
        return punctuation_pause
    elif ends_with_string(text, "!"):
        return exclamation_pause
    elif ends_with_string(text, "?"):
        return question_pause
    else:
        return unknown_sentence_detection_pause

def preprocess_text(text):
    text = text.lstrip() # Remove leading whitespaces
    if text.startswith("..."): #  Remove starting ellipses if present
        text = text[3:]
    text = text.lstrip() # Remove any leading whitespaces again after ellipses removal
    if text:
        text = text[0].upper() + text[1:] # Uppercase the first letter

    return text

def strip_ending_punctuation(text):
    """Remove trailing periods and ellipses from text."""
    text = text.rstrip()
    for char in sentence_end_marks:
        text = text.rstrip(char)
    return text

def find_matching_texts(texts_without_punctuation):
    """
    Find entries where text_without_punctuation matches the last entry,
    going backwards until the first non-match is found.
    
    Args:
        texts_without_punctuation: List of tuples (original_text, stripped_text)
        
    Returns:
        List of tuples (original_text, stripped_text) matching the last entry's stripped text,
        stopping at the first non-match
    """
    if not texts_without_punctuation:
        return []
    
    # Get the stripped text from the last entry
    last_stripped_text = texts_without_punctuation[-1][1]
    
    matching_entries = []
    
    # Iterate through the list backwards
    for entry in reversed(texts_without_punctuation):
        original_text, stripped_text = entry
        
        # If we find a non-match, stop
        if stripped_text != last_stripped_text:
            break
            
        # Add the matching entry to our results
        matching_entries.append((original_text, stripped_text))
    
    # Reverse the results to maintain original order
    matching_entries.reverse()
    
    return matching_entries

def interpolate_detection(prob):
    # Clamp probability between 0.0 and 1.0 just in case
    p = max(0.0, min(prob, 1.0))
    # If exactly at an anchor point
    for ap_p, ap_val in anchor_points:
        if abs(ap_p - p) < 1e-9:
            return ap_val

    # Find where p fits
    for i in range(len(anchor_points) - 1):
        p1, v1 = anchor_points[i]
        p2, v2 = anchor_points[i+1]
        if p1 <= p <= p2:
            # Linear interpolation
            ratio = (p - p1) / (p2 - p1)
            return v1 + ratio * (v2 - v1)

    # Should never reach here if anchor_points cover [0,1]
    return 4.0

class TurnDetection:
    """
    Handles the turn detection logic.
    """

    def __init__(
        self,
        on_new_waiting_time: callable,
        local: bool = False,
    ) -> None:
        
        model_dir = model_dir_local if local else model_dir_cloud

        self.on_new_waiting_time = on_new_waiting_time

        self.current_waiting_time = -1
        self.text_time_deque = collections.deque()
        self.texts_without_punctuation = []

        self.text_queue: queue.Queue[str] = queue.Queue()
        self.text_worker = threading.Thread(
            target=self._text_worker,
            daemon=True
        )
        self.text_worker.start()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(model_dir)
        self.classification_model = transformers.DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.classification_model.to(self.device)
        self.classification_model.eval()
        self.max_length = 128
    
    def suggest_time(
            self,
            time: float,
            text: str = None
        ) -> None:
        if time == self.current_waiting_time:
            return

        self.current_waiting_time = time

        if self.on_new_waiting_time:
            self.on_new_waiting_time(time, text)

    def get_completion_probability(
        self, 
        sentence: str
    ) -> float:
        """
        Return the probability that the sentence is complete.
        """
        # If there's no cache yet, create one
        if not hasattr(self, "_completion_probability_cache"):
            self._completion_probability_cache = {}
        
        # Return from cache if we already have it
        if sentence in self._completion_probability_cache:
            return self._completion_probability_cache[sentence]
        
        import torch
        import torch.nn.functional as F

        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.classification_model(**inputs)

        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()
        prob_complete = probabilities[1]

        # Store in cache before returning
        self._completion_probability_cache[sentence] = prob_complete
        return prob_complete

    def _text_worker(
        self
    ) -> None:
        """
        Worker thread for turn detection.
        """
        while True:
            # Drain the queue, keeping only the last text.
            text = None
            while True:
                try:
                    text = self.text_queue.get_nowait()
                except queue.Empty:
                    break
            if not text:
                time.sleep(0.001)
                continue

            #text = self.text_queue.get()
            logger.info(f"👂 Starting to calculate waiting time for txt: {text}")
            text = preprocess_text(text)

            current_time = time.time()
            self.text_time_deque.append((current_time, text))
            text_without_punctuation = strip_ending_punctuation(text)
            self.texts_without_punctuation.append((text, text_without_punctuation))
            matches = find_matching_texts(self.texts_without_punctuation)

            added_pauses = 0
            contains_ellipses = False
            for i, match in enumerate(matches):
                same_text, stripped_punctuation = match
                suggested_pause = get_suggested_whisper_pause(same_text)
                added_pauses += suggested_pause
                if ends_with_string(same_text, "..."):
                    contains_ellipses = True

            avg_pause = added_pauses / len(matches) if len(matches) > 0 else 0
            suggested_pause = avg_pause

            self.prev_text = text
            import string
            transtext = text.translate(str.maketrans('', '', string.punctuation))

            cleaned_for_model = re.sub(r'[^a-zA-Z]+$', '', transtext)

            prob_complete = self.get_completion_probability(cleaned_for_model)

            new_detection = interpolate_detection(prob_complete)

            pause = (new_detection + suggested_pause) * detection_speed

            if contains_ellipses:
                pause += 0.2

            logger.info(f"👂 Suggest time for txt: {text}")
            self.suggest_time(pause, text)

    def calculate_waiting_time(
            self,
            text: str) -> None:
        """
        Preprocess the given text for display.
        """
        logger.info(f"👂 Put txt to text_queue for turn detection waiting time calc: {text}")
        self.text_queue.put(text)

    def reset(self):
        self.text_time_deque.clear()
        self.texts_without_punctuation.clear()
        self.current_waiting_time = -1
        self.texts_without_punctuation = []