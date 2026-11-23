# GUITAR GUIDANCE

## Overview
This project aims to leverage machine learning models to predict chords played from a guitar. Currently, there are only 8 chords implemented: A major, A minor, C major, D major, D minor, E major, E minor, and G major. For future plans, the rest of the open chords will be added along with suspended chords.

## How to Use
To install this system, the repository must be pulled. In the meantime, Docker Desktop is required to fully deploy this app; however, there are future plans to put this on a remote server as a web app. Once the repository has been pulled, using docker desktop build the image using docker compose build and docker compose up. This should bring the system online in your local host at port 7860. Once the application is open, ensure you have an **audio interface** to plug your guitar in with your PC. The model has been trained using only an audio interface; open audio has not yet been tested thus is not reliable. Select your input device using the drop down menu in the app and hit start recording. While you play your chords should display clearly on the screen.

## Model Specifics
The model that is used for this system is a **logistic regression model** meaning it is very lightweight. It was trained using an Ibanez Jem Jr guitar with a Boss Katana 100 Gen 3 as an amplifier/audio interface. Using fixed audio, this model demonstrated a 100% accuracy on the training set, and a 99% accuracy on the test set. Using a holdout set, this model performed comparatively using distorted tones and other effects that may change the audio. The biggest struggle was with different chord voicings. When transitioning to live audio, the accuracy dropped slightly (featuring roughly around an 80% accuracy) with the chord it struggles the most to identify is A minor. 
