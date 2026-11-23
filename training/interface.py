
import time
import random
import gradio as gr
import soundfile as sf
import threading
import os
import numpy as np
import sounddevice as sd
import pickle
import librosa
import pandas as pd

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


# --------------------------------------------------
# Microphone capture manager
# --------------------------------------------------

class AudioCapture:

    def __init__(self):
        self.stream = None
        self.lock = threading.Lock()

        self.buffer = None
        self.sample_rate = None
        self.capacity = None

        self.write_position = 0
        self.total_samples = 0

        self.next_window_end = 0
        self.window_count = 0

        self.active = False
        self.last_error = None

    def start(self, device_index):

        self.stop()

        # Get the selected microphone's default sample rate
        device_info = sd.query_devices(
            device_index,
            "input"
        )

        sample_rate = int(device_info["default_samplerate"])

        capacity = int(WINDOW_SECONDS * sample_rate)

        with self.lock:

            self.sample_rate = sample_rate
            self.capacity = capacity

            # Fixed-size rolling buffer
            self.buffer = np.zeros(
                capacity,
                dtype=np.float32
            )

            self.write_position = 0
            self.total_samples = 0

            self.next_window_end = capacity
            self.window_count = 0

            self.last_error = None
            self.active = True

        try:

            self.stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=sample_rate,
                dtype="float32",
                blocksize=0,
                callback=self.audio_callback
            )

            self.stream.start()

        except Exception:
            with self.lock:
                self.active = False
            raise

        return sample_rate

    def audio_callback(self, indata, frames, time_info, status):

        if status:
            self.last_error = str(status)

        # Extract mono audio
        samples = indata[:, 0]

        with self.lock:

            if not self.active:
                return

            n = len(samples)

            # If one callback contains more than the buffer,
            # retain only its most recent samples.
            if n >= self.capacity:

                self.buffer[:] = samples[-self.capacity:]
                self.write_position = 0

            else:

                end_position = self.write_position + n

                # Handle circular-buffer wraparound
                if end_position <= self.capacity:

                    self.buffer[
                        self.write_position:end_position
                    ] = samples

                else:

                    first_part = self.capacity - self.write_position

                    self.buffer[self.write_position:] = samples[:first_part]

                    self.buffer[:end_position - self.capacity] = samples[first_part:]

                self.write_position = end_position % self.capacity

            self.total_samples += n

    def get_next_window(self):

        with self.lock:

            if not self.active:
                return None

            if self.total_samples < self.next_window_end:
                return None

            # If polling was delayed, skip obsolete windows
            # rather than returning duplicate audio.
            windows_due = (
                1 + (self.total_samples - self.next_window_end)
                // int(HOP_SECONDS * self.sample_rate)
            )

            self.next_window_end += (
                windows_due * int(HOP_SECONDS * self.sample_rate)
            )

            # Reconstruct chronological audio from circular buffer
            if self.write_position == 0:

                audio = self.buffer.copy()

            else:

                audio = np.concatenate((
                    self.buffer[self.write_position:],
                    self.buffer[:self.write_position]
                ))

            self.window_count += 1

            return (
                self.sample_rate,
                audio,
                self.window_count,
                self.last_error
            )

    def stop(self):

        with self.lock:
            self.active = False

        if self.stream is not None:

            try:
                self.stream.stop()
                self.stream.close()

            except Exception:
                pass

            self.stream = None

    def is_active(self):

        with self.lock:
            return self.active


capture = AudioCapture()


def get_input_devices():

    devices = sd.query_devices()

    choices = []

    for index, device in enumerate(devices):

        if device["max_input_channels"] > 0:

            name = device["name"]

            label = f"{name} (Device {index})"

            choices.append((label, index))

    # Find the system's default input device
    default_device = sd.default.device[0]

    available_indices = [value for _, value in choices]

    if default_device in available_indices:
        selected = default_device

    elif choices:
        selected = choices[0][1]

    else:
        selected = None

    return choices, selected


def refresh_devices():

    choices, selected = get_input_devices()

    return gr.update(
        choices=choices,
        value=selected
    )


def start_recording(device_index):

    if device_index is None:

        return (
            "No microphone selected.",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True)
        )

    try:

        sample_rate = capture.start(device_index)

        return (
            f"Recording... ({sample_rate} Hz)",
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=False)
        )

    except Exception as e:

        return (
            f"Microphone error: {str(e)}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True)
        )


def stop_recording():

    capture.stop()

    return (
        "Recording stopped.",
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=True)
    )

def process_window():
    print("Checking for audio window...")

    result = capture.get_next_window()

    if result is None:
        return None, "Waiting for audio...", None

    sample_rate, audio, window_number, error = result

    try:
        predicted_chord = predictChord(audio, sample_rate)

        print(f"Prediction: {predicted_chord}")

        if error:
            print(f"Audio warning: {error}")

        return predicted_chord, "Prediction successful", window_number

    except Exception as e:
        print(f"PREDICTION ERROR: {e}")
        return None, f"Prediction error: {str(e)}", window_number


def extractFeatures(path, sr):
    y, sr = librosa.load(path, sr=sr,  mono = True)
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

    os.makedirs("TestAudio", exist_ok=True)

    # Save the audio window
    sf.write(
        "TestAudio/window.wav",
        audio,
        sample_rate
    )

    # Extract features
    features = extractFeatures("TestAudio/window.wav", sample_rate)

    # Convert features into a DataFrame
    features_df = pd.DataFrame([features])

    # Match the feature order used during training, if available
    if hasattr(model, "feature_names_in_"):
        features_df = features_df.reindex(
            columns=model.feature_names_in_
        )

    # Predict chord
    prediction = model.predict(features_df)
    probabilities = model.predict_proba(features_df)

    confidence = max(probabilities[0]) * 100

    if confidence < 45:
        return "Unknown Chord"

    return prediction[0]

    

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
# Session logic
# --------------------------------------------------

def start_session():
    """Start listening for chords."""

    return (
        True,
        time.time(),
        -1,
        "Listening for Chords...",
        "Identified: **—**",
        "—",
        gr.update(interactive=False),
        gr.update(interactive=True),
        "",
        []
    )


def stop_session():
    """Stop listening and reset the prediction."""

    return (
        False,
        None,
        -1,
        "Not currently listening.",
        "Identified: **—**",
        "—",
        gr.update(interactive=True),
        gr.update(interactive=False),
        "Listening stopped.",
        []
    )


def start_listening(device_index):
    """Start microphone capture and the existing chord-listening session."""

    recording_status, start_update, stop_update, device_update = start_recording(
        device_index
    )

    if not capture.is_active():
        # Microphone startup failed; leave the chord session stopped.
        return (
            recording_status,
            device_update,
            gr.update(),
            gr.update(),
            gr.update(),
            "Not currently listening.",
            gr.update(),
            gr.update(),
            start_update,
            stop_update,
            gr.update(),
            gr.update()
        )

    session_values = start_session()

    return (
        recording_status,
        device_update,
        *session_values
    )


def stop_listening():
    """Stop microphone capture and reset the existing chord session."""

    recording_status, _, _, _ = stop_recording()
    session_values = stop_session()

    return (
        recording_status,
        gr.update(interactive=True),
        *session_values
    )


def update_chord(active, start_time, last_update):

    if not active or start_time is None:
        return (
            "Not currently listening.",
            gr.update(),
            gr.update(),
            last_update
        )

    predicted, status, window_number = process_window()

    if predicted is None:
        print(status)

        return (
            "Listening for Chords...",
            gr.update(),
            gr.update(),
            last_update
        )

    chord_name = CHORD_NAMES.get(str(predicted), str(predicted))

    print(f"Updating interface with: {chord_name}")

    return (
        "Listening for Chords...",
        f"Identified: **{chord_name}**",
        str(predicted),
        window_number
    )


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

#identified-chord p {
    font-size: 25px !important;
    font-weight: 650 !important;
    text-align: center !important;
    color: #20c4cf !important;
    margin: 15px 0 !important;
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


/* Audio input device controls */

#device-controls {
    width: 100% !important;
    justify-content: center !important;
    align-items: center !important;
    margin: 18px auto 0 auto !important;
}

#device-dropdown {
    width: 300px !important;
    max-width: 90% !important;
    margin: 0 auto !important;
}

#device-dropdown label {
    color: #d0d2d9 !important;
    font-weight: 600 !important;
}

#device-dropdown input,
#device-dropdown [role="combobox"] {
    background: #343843 !important;
    border: 1px solid #555b68 !important;
    color: #f0f0f0 !important;
    border-radius: 8px !important;
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


/* Disabled buttons */

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

        identified_display = gr.Markdown(
            "Identified: **—**",
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
        # Listening controls
        # ------------------------------------------

        with gr.Row():

            start_button = gr.Button(
                "Start Listening",
                elem_id="start-button"
            )

            stop_button = gr.Button(
                "Stop Listening",
                elem_id="stop-button",
                interactive=False
            )


        # ------------------------------------------
        # Audio input device selection
        # ------------------------------------------

        with gr.Row(elem_id="device-controls"):

            device_dropdown = gr.Dropdown(
                choices=[],
                value=None,
                label="Audio Input Device",
                interactive=True,
                filterable=True,
                elem_id="device-dropdown"
            )

        recording_status = gr.Markdown(
            "Microphone ready.",
            elem_id="recording-status"
        )



    # ----------------------------------------------
    # Start listening
    # ----------------------------------------------

    start_button.click(
        fn=start_listening,
        inputs=[device_dropdown],
        outputs=[
            recording_status,
            device_dropdown,
            session_active,
            session_start,
            last_update,
            listening_status,
            identified_display,
            detected_chord,
            start_button,
            stop_button,
            feedback_message,
            feedback_history
        ]
    )

    # Refresh the available microphones whenever the app is opened.
    demo.load(
        fn=refresh_devices,
        inputs=[],
        outputs=[device_dropdown]
    )


    # ----------------------------------------------
    # Stop listening
    # ----------------------------------------------

    stop_button.click(
        fn=stop_listening,
        inputs=[],
        outputs=[
            recording_status,
            device_dropdown,
            session_active,
            session_start,
            last_update,
            listening_status,
            identified_display,
            detected_chord,
            start_button,
            stop_button,
            feedback_message,
            feedback_history
        ]
    )


    # ----------------------------------------------
    # Simulated live chord detection
    # ----------------------------------------------

    refresh_timer = gr.Timer(1.0)

    refresh_timer.tick(
        fn=update_chord,
        inputs=[
            session_active,
            session_start,
            last_update
        ],
        outputs=[
            listening_status,
            identified_display,
            detected_chord,
            last_update
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
