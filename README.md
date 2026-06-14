# CS898BA Project 1: OpenCV Setup

## Project Overview
This repository contains the foundational setup for Project 1. The goal is to establish a version-controlled Python codebase, verify the OpenCV environment, and maintain an audit log of AI assistance.

## Code Explanation
The `hello.py` script is designed to test the environment setup:
* `import cv2`: Loads the OpenCV library.
* `print("Hello World!")`: A standard test to ensure Python is running correctly.
* `print(f"OpenCV version: {cv2.__version__}")`: Accesses and displays the currently installed version of OpenCV to confirm the package is available to the script.

## Setup Instructions
1.  Ensure Python 3 is installed on your machine.
2.  Open your terminal or command prompt.
3.  Install the necessary computer vision library by running:
    `pip install opencv-python`

## Execution Steps
To run the script, open your terminal, navigate to the folder containing this repository, and execute:
`python hello.py`

## Expected Results
When run successfully, the terminal will output the greeting followed by your specific version of OpenCV. For example:
> Hello World!
> OpenCV version: 4.9.0

