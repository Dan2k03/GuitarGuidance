
import time
import random
import gradio as gr
import threading
import os
import numpy as np
import pickle
import librosa
import pandas as pd
from collections import Counter
from prometheus_client import Counter as PrometheusCounter, Gauge, Histogram, start_http_server

# --------------------------------------------------
# Prometheus metrics
# --------------------------------------------------

# These metrics describe model inference, not only the smoothed UI output.
# The chord label has a small, fixed set of possible values.
PREDICTIONS_TOTAL = PrometheusCounter(
    "guitar_guidance_predictions_total",
    "Number of chord predictions made by the model",
    ["chord"]
)

PREDICTION_CONFIDENCE = Histogram(
    "guitar_guidance_prediction_confidence_percent",
    "Distribution of model prediction confidence, in percent",
    buckets=(0, 20, 40, 45, 50, 60, 70, 80, 90, 95, 99, 100)
)

LAST_PREDICTION_CONFIDENCE = Gauge(
    "guitar_guidance_last_prediction_confidence_percent",
    "Confidence of the most recent model prediction, in percent"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------
with open("finalModel.pkl", "rb") as f:
        model = pickle.load(f)

print(type(model))
print(hasattr(model, "predict_proba"))

WINDOW_SECONDS = 0.75
OVERLAP_SECONDS = 0.6

# Time between predictions:
HOP_SECONDS = WINDOW_SECONDS - OVERLAP_SECONDS

# Stabilize the displayed chord across several consecutive predictions.
PREDICTION_HISTORY_SIZE = 5
MINIMUM_VOTES = 3

# Ignore very quiet windows so background noise does not trigger predictions.
# This is a normalized RMS threshold; adjust if a quiet guitar signal is ignored.
SILENCE_RMS_THRESHOLD = 0.005


# --------------------------------------------------
# Browser microphone audio processing
# --------------------------------------------------

# Each browser session maintains its own rolling audio buffer in gr.State.
# The browser captures the user's microphone; the Docker host does not need
# direct access to the user's audio hardware.

def process_browser_audio(audio_buffer, prediction_history, new_chunk, current_chord):
    """Process browser audio and smooth predictions before updating the display."""

    if prediction_history is None:
        prediction_history = []

    if new_chunk is None:
        return audio_buffer, prediction_history, gr.skip(), gr.skip(), gr.skip()

    try:
        sample_rate, chunk = new_chunk
        chunk = np.asarray(chunk)

        if chunk.size == 0:
            return audio_buffer, prediction_history, gr.skip(), gr.skip(), gr.skip()

        # Convert stereo input to mono.
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)

        # Ensure floating-point audio in the range expected by librosa.
        if np.issubdtype(chunk.dtype, np.integer):
            max_value = np.iinfo(chunk.dtype).max
            chunk = chunk.astype(np.float32) / max_value
        else:
            chunk = chunk.astype(np.float32)

        # Start a fresh buffer if this is the first chunk or the sample rate changed.
        if audio_buffer is None or audio_buffer[0] != int(sample_rate):
            combined = chunk
        else:
            combined = np.concatenate((audio_buffer[1], chunk))

        window_samples = int(WINDOW_SECONDS * sample_rate)

        # Wait until we have enough audio for the model's 0.75-second window.
        if len(combined) < window_samples:
            return (int(sample_rate), combined), prediction_history, gr.skip(), gr.skip(), gr.skip()

        # Keep only the most recent window.
        window = combined[-window_samples:]
        updated_buffer = (int(sample_rate), window)

        # Ignore quiet windows instead of allowing room noise to create guesses.
        window_rms = float(np.sqrt(np.mean(np.square(window))))
        if window_rms < SILENCE_RMS_THRESHOLD:
            return updated_buffer, prediction_history, gr.skip(), gr.skip(), gr.skip()

        predicted_chord, confidence = predictChord(window, int(sample_rate))
        predicted_chord = str(predicted_chord)

        # Retain only the five most recent predictions.
        prediction_history.append(predicted_chord)
        prediction_history = prediction_history[-PREDICTION_HISTORY_SIZE:]

        # Require at least three matching predictions before changing the display.
        counts = Counter(prediction_history)
        most_common_chord, vote_count = counts.most_common(1)[0]

        if vote_count < MINIMUM_VOTES:
            return updated_buffer, prediction_history, gr.skip(), gr.skip(), gr.skip()

        # Only update the visible components when the stable prediction changes.
        if most_common_chord == current_chord:
            return updated_buffer, prediction_history, gr.skip(), gr.skip(), gr.skip()

        chord_name = CHORD_NAMES.get(most_common_chord, most_common_chord)
        return (
            updated_buffer,
            prediction_history,
            gr.skip(),
            f"Identified: **{chord_name}**",
            most_common_chord
        )

    except Exception as e:
        print(f"BROWSER AUDIO PROCESSING ERROR: {e}")
        return audio_buffer, prediction_history, f"Audio processing error: {str(e)}", gr.skip(), gr.skip()


def browser_recording_started():
    return "Microphone active. Play a chord when you're ready."


def browser_recording_stopped():
    return None, [], "Recording stopped.", "Identified: **—**", "—"


def extractFeatures(y, sr):
    y = np.asarray(y, dtype=np.float32)
    y, _ = librosa.effects.trim(y)

    feats = {}

    #Most useful feature
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma_mean_{i}"] = chroma[i].mean()
        feats[f"chroma_std_{i}"] = chroma[i].std()

    timbre = librosa.feature.mfcc(y=y, sr=sr)
    for i in range(13):
        feats[f"timbre_mean_{i}"] = timbre[i].mean()
        feats[f"timbre_std_{i}"] = timbre[i].std()


    feats["spec_centroid"] = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    feats["spec_bandwidth"] = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    feats["spec_rolloff"] = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    feats["zcr"] = librosa.feature.zero_crossing_rate(y).mean()
    feats["rms"] = librosa.feature.rms(y=y).mean()
    return feats


def predictChord(audio, sample_rate):

    # Extract features directly from the browser-provided audio window.
    features = extractFeatures(audio, sample_rate)

    # Convert features into a DataFrame
    features_df = pd.DataFrame([features])

    # Match the feature order used during training, if available
    if hasattr(model, "feature_names_in_"):
        features_df = features_df.reindex(
            columns=model.feature_names_in_
        )

    # Predict chord and calculate the model's highest class probability.
    prediction = model.predict(features_df)
    probabilities = model.predict_proba(features_df)

    confidence = float(np.max(probabilities[0]) * 100)
    predicted_chord = str(prediction[0])

    if confidence < 45:
        predicted_chord = "Unknown Chord"

    # Record every inference, including low-confidence/unknown predictions.
    PREDICTIONS_TOTAL.labels(chord=predicted_chord).inc()
    PREDICTION_CONFIDENCE.observe(confidence)
    LAST_PREDICTION_CONFIDENCE.set(confidence)

    return predicted_chord, confidence

    

# --------------------------------------------------
# Chord definitions
# --------------------------------------------------

CHORDS = ["Am", "Cmaj", "Dmaj", "Em", "Fmaj", "Gmaj", "Amaj", "Dm", "Emaj"]

CHORD_NAMES = {
    "Am": "A minor",
    "Cmaj": "C major",
    "Dmaj": "D major",
    "Em": "E minor",
    "Fmaj": "F major",
    "Gmaj": "G major",
    "Amaj": "A major",
    "Dm": "D minor",
    "Emaj": "E major"
}


# --------------------------------------------------
# User feedback
# --------------------------------------------------

def submit_feedback(answer, chord, history):
    """Record whether the chord prediction was accurate."""

    if chord == "—":
        return "Please play a chord first.", history

    history = history + [{
        "Chord": chord,
        "Accurate": answer,
        "Time": time.strftime("%H:%M:%S")
    }]

    if answer == "Yes":
        message = "Thanks for your feedback!"
    else:
        message = "Thanks! We'll use this to improve predictions."

    return message, history


# --------------------------------------------------
# Styling
# --------------------------------------------------

CSS = """

/* Entire page */

body {
    background: #17191f !important;
}

.gradio-container {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 auto !important;
    background: #17191f !important;
    color: #f0f0f0 !important;
    padding: 20px !important;
}

footer {
    display: none !important;
}


/* Main panel */

#main-panel {
    width: 600px !important;
    max-width: calc(100vw - 40px) !important;
    min-width: 0 !important;
    margin: 10px auto !important;
    padding: 30px 25px !important;
    box-sizing: border-box !important;

    background: #242730 !important;
    border: 2px solid #414550 !important;
    border-radius: 8px !important;

    gap: 12px !important;
}


/* General text */

#main-panel,
#main-panel p,
#main-panel h1,
#main-panel h2,
#main-panel label {
    color: #f0f0f0 !important;
}


/* Title */

#page-title h1 {
    font-size: 36px !important;
    font-weight: 750 !important;
    text-align: center !important;
    color: #ffffff !important;
    margin: 5px 0 25px 0 !important;
}


/* Listening message */

#listening-status p {
    font-size: 25px !important;
    font-weight: 600 !important;
    text-align: center !important;
    color: #b0b3bd !important;
    margin: 10px 0 !important;
}


/* Identified chord */

#identified-chord textarea,
#identified-chord input {
    font-size: 25px !important;
    font-weight: 650 !important;
    text-align: center !important;
    color: #20c4cf !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    resize: none !important;
    padding: 8px 0 !important;
}


/* Feedback question */

#feedback-question p {
    font-size: 19px !important;
    font-weight: 600 !important;
    text-align: center !important;
    color: #d0d2d9 !important;
    margin-top: 15px !important;
}


/* Feedback buttons */

#yes-button, #no-button {
    background: #343843 !important;
    border: 1px solid #555b68 !important;
    border-radius: 9px !important;
    color: #f0f0f0 !important;
    font-size: 17px !important;
    font-weight: 650 !important;
    min-height: 45px !important;
}

#yes-button:hover, #no-button:hover {
    background: #414957 !important;
    border-color: #20c4cf !important;
}


/* Feedback response */

#feedback-message p {
    color: #20c4cf !important;
    font-size: 15px !important;
    text-align: center !important;
    font-weight: 600 !important;
}


/* Start listening */

#start-button {
    background: #087f8c !important;
    border: 1px solid #20c4cf !important;
    border-radius: 10px !important;
    color: white !important;
    font-size: 17px !important;
    font-weight: 650 !important;
    min-height: 55px !important;
}

#start-button:hover {
    background: #0999a7 !important;
}


/* Stop listening */

#stop-button {
    background: #343843 !important;
    border: 1px solid #555b68 !important;
    border-radius: 10px !important;
    color: white !important;
    font-size: 17px !important;
    font-weight: 650 !important;
    min-height: 55px !important;
}

#stop-button:hover {
    background: #414957 !important;
}


/* Microphone recording status */

#recording-status p {
    color: #20c4cf !important;
    font-size: 14px !important;
    text-align: center !important;
    min-height: 20px !important;
    max-width: 390px !important;
    margin: 4px auto 0 auto !important;
}


#start-button:disabled,
#stop-button:disabled {
    opacity: 0.4 !important;
}

"""


# --------------------------------------------------
# Gradio interface
# --------------------------------------------------

with gr.Blocks(
    title="Guitar Guidance - Free Chord"
) as demo:

    # ----------------------------------------------
    # Session state
    # ----------------------------------------------

    session_active = gr.State(False)
    session_start = gr.State(None)
    last_update = gr.State(-1)

    detected_chord = gr.State("—")
    feedback_history = gr.State([])


    # ----------------------------------------------
    # Main interface
    # ----------------------------------------------

    with gr.Column(elem_id="main-panel"):

        gr.Markdown(
            "# Free Chord",
            elem_id="page-title"
        )

        listening_status = gr.Markdown(
            "Not currently listening.",
            elem_id="listening-status"
        )

        # A Textbox avoids Markdown re-rendering (which can look like
        # the chord label is blinking whenever a prediction changes).
        identified_display = gr.Textbox(
            value="Identified: —",
            label=None,
            show_label=False,
            interactive=False,
            container=False,
            elem_id="identified-chord"
        )


        # ------------------------------------------
        # User feedback
        # ------------------------------------------

        gr.Markdown(
            "Was this prediction accurate?",
            elem_id="feedback-question"
        )

        with gr.Row():

            yes_button = gr.Button(
                "Yes",
                elem_id="yes-button"
            )

            no_button = gr.Button(
                "No",
                elem_id="no-button"
            )

        feedback_message = gr.Markdown(
            "",
            elem_id="feedback-message"
        )


        # ------------------------------------------
        # Browser microphone input
        # ------------------------------------------

        microphone_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            streaming=True,
            label="Microphone / Audio Interface",
            elem_id="browser-microphone"
        )

        recording_status = gr.Markdown(
            "Select the microphone control above and allow browser access.",
            elem_id="recording-status"
        )

        audio_buffer = gr.State(None)
    prediction_history = gr.State([])



    # ----------------------------------------------
    # Browser microphone streaming
    # ----------------------------------------------

    microphone_input.start_recording(
        fn=browser_recording_started,
        inputs=[],
        outputs=[recording_status]
    )

    microphone_input.stream(
        fn=process_browser_audio,
        inputs=[audio_buffer, prediction_history, microphone_input, detected_chord],
        outputs=[
            audio_buffer,
            prediction_history,
            recording_status,
            identified_display,
            detected_chord
        ],
        stream_every=HOP_SECONDS,
        time_limit=3600,
        concurrency_limit=1
    )

    microphone_input.stop_recording(
        fn=browser_recording_stopped,
        inputs=[],
        outputs=[
            audio_buffer,
            prediction_history,
            recording_status,
            identified_display,
            detected_chord
        ]
    )


    # ----------------------------------------------
    # Feedback submission
    # ----------------------------------------------

    yes_button.click(
        fn=lambda chord, history: submit_feedback(
            "Yes", chord, history
        ),
        inputs=[
            detected_chord,
            feedback_history
        ],
        outputs=[
            feedback_message,
            feedback_history
        ]
    )

    no_button.click(
        fn=lambda chord, history: submit_feedback(
            "No", chord, history
        ),
        inputs=[
            detected_chord,
            feedback_history
        ],
        outputs=[
            feedback_message,
            feedback_history
        ]
    )


# --------------------------------------------------
# Launch application
# --------------------------------------------------

if __name__ == "__main__":
    # Prometheus scrapes this endpoint from the Gradio container on port 8000.
    start_http_server(8000, addr="0.0.0.0")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="slate",
            neutral_hue="slate"
        ),
        css=CSS
    )
