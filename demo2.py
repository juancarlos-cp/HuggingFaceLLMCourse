from transformers import pipeline
import gradio as gr

asr_model = pipeline("automatic-speech-recognition")

def transcribe_audio(audio):
    import numpy as np

    if audio is None:
        return "No audio input detected."

    sampling_rate, audio_array = audio
    audio_array = audio_array.astype(np.float32)

    # Reshape to [samples, 1] if it's 1D
    if audio_array.ndim == 1:
        audio_array = np.expand_dims(audio_array, axis=1)

    result = asr_model({"array": audio_array.squeeze(), "sampling_rate": sampling_rate})
    return result["text"]

mic = gr.Audio(sources=["microphone"], type="numpy", label="Speak here...")

gr.Interface(
    fn=transcribe_audio,
    inputs=mic,
    outputs="text"
).launch()
