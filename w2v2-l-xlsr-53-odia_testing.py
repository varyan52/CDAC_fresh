# Imports
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


# Loading
test_dataset = load_dataset("common_voice", "or", split="test[:2%]")
processor = Wav2Vec2Processor.from_pretrained("anuragshas/wav2vec2-large-xlsr-53-odia")
model = Wav2Vec2ForCTC.from_pretrained("anuragshas/wav2vec2-large-xlsr-53-odia")
resampler = torchaudio.transforms.Resample(48_000, 16_000)


# Preprocessing
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["or_test_0/or_test_0/common_voice_or_21820802.mp3"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch
test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"][:2], sampling_rate=16_000, return_tensors="pt", padding=True)


# Prediction / Transcription
with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
predicted_ids = torch.argmax(logits, dim=-1)
print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset["sentence"][:2])