
import time
import threading
import numpy as np
import sounddevice as sd
import gradio as gr


# --------------------------------------------------
# Configuration
# --------------------------------------------------

WINDOW_SECONDS = 3
OVERLAP_SECONDS = 1

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


# One capture manager for this local test application
capture = AudioCapture()


# --------------------------------------------------
# Audio device selection
# --------------------------------------------------

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


# --------------------------------------------------
# Start and stop recording
# --------------------------------------------------

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


# --------------------------------------------------
# Retrieve sliding windows
# --------------------------------------------------

def process_window():

    result = capture.get_next_window()

    if result is None:

        if capture.is_active():

            return gr.update(), "Waiting for audio..."

        return gr.update(), "Not currently recording."

    sample_rate, audio, window_number, error = result

    duration = len(audio) / sample_rate

    status = (
        f"Window #{window_number} | "
        f"{duration:.1f} seconds | "
        f"{sample_rate} Hz"
    )

    if error:
        status += f" | Audio warning: {error}"

    # Return the audio to Gradio for playback
    return (
        (sample_rate, audio),
        status
    )


# --------------------------------------------------
# Styling
# --------------------------------------------------

CSS = """

body {
    background: #17191f !important;
}

.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    background: #17191f !important;
    color: #f0f0f0 !important;
}

footer {
    display: none !important;
}

#main-panel {
    background: #242730 !important;
    border: 2px solid #414550 !important;
    border-radius: 10px !important;
    padding: 25px !important;
}

#page-title h1 {
    text-align: center !important;
    color: white !important;
}

#status {
    color: #20c4cf !important;
}

#start-button {
    background: #087f8c !important;
    color: white !important;
}

#stop-button {
    background: #343843 !important;
    color: white !important;
}

"""


# --------------------------------------------------
# Gradio interface
# --------------------------------------------------

devices, default_device = get_input_devices()

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css=CSS,
    title="Guitar Guidance - Audio Test"
) as demo:

    with gr.Column(elem_id="main-panel"):

        gr.Markdown(
            "# Microphone & Audio Testing",
            elem_id="page-title"
        )

        gr.Markdown(
            "Select your microphone and start recording."
        )

        # Device selection
        with gr.Row():

            device_dropdown = gr.Dropdown(
                choices=devices,
                value=default_device,
                label="Microphone Device",
                scale=5
            )

            refresh_button = gr.Button(
                "Refresh Devices",
                scale=1
            )

        # Recording controls
        with gr.Row():

            start_button = gr.Button(
                "Start Recording",
                elem_id="start-button"
            )

            stop_button = gr.Button(
                "Stop Recording",
                elem_id="stop-button",
                interactive=False
            )

        status_display = gr.Markdown(
            "Not currently recording.",
            elem_id="status"
        )

        # Latest extracted audio window
        gr.Markdown("### Latest Audio Window")

        window_audio = gr.Audio(
            label="5-second recording",
            interactive=False
        )

        window_status = gr.Markdown(
            "Waiting for first window..."
        )

        gr.Markdown(
            f"""
            **Window size:** {WINDOW_SECONDS} seconds

            **Overlap:** {OVERLAP_SECONDS} second

            **Prediction interval:** {HOP_SECONDS} seconds
            """
        )

    # Refresh available microphones
    refresh_button.click(
        fn=refresh_devices,
        outputs=[device_dropdown]
    )

    # Start recording
    start_button.click(
        fn=start_recording,
        inputs=[device_dropdown],
        outputs=[
            status_display,
            start_button,
            stop_button,
            device_dropdown
        ]
    )

    # Stop recording
    stop_button.click(
        fn=stop_recording,
        outputs=[
            status_display,
            start_button,
            stop_button,
            device_dropdown
        ]
    )

    # Check for new audio windows
    timer = gr.Timer(0.25)

    timer.tick(
        fn=process_window,
        outputs=[
            window_audio,
            window_status
        ]
    )


# --------------------------------------------------
# Launch
# --------------------------------------------------

if __name__ == "__main__":
    demo.launch()
