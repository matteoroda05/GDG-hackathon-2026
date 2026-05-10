# GDG-hackathon-2026

Eco-island rubbish detection for the GDG hackathon, built on Luxonis OAK devices.

## Project Goals

- Identify a waste item placed in the detection area and classify it into the target bin classes.
- Detect wrong-bin items and surface them clearly to the operator.
- Estimate bin fullness from depth to signal when it should be emptied.
- Provide a web dashboard for the live demo and judging flow.

## Target Classes

- Plastic
- Metal
- Paper
- Glass
- Organic
- Generic

## Current State

- Live detection pipeline with stereo depth and a browser dashboard.
- Wrong-bin highlighting and bin fullness estimation.
- Support for custom model archives with labels read from the model config.

## Repo Layout

- [base-app](base-app): runtime application, device pipeline, and dashboard.
- [training](training): dataset preparation and training scripts.
- [tests](tests): classification and depth estimation tests for mapping and dashboard behavior.

## Demo Notes

- The runtime and setup commands live in [base-app/README.md](base-app/README.md).
- The demo assumes the physical eco-island system handles user identification and bin actuation.