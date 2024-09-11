import os
from pathlib import Path
import logging

# Configure logging for better visibility of actions
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# List of files required in the project
list_of_files = [
    "src/__init__.py",
    "src/lip_reading_model.py",  # File for LipReadingModel class
    "src/video_processor.py",    # File for VideoProcessor class
    "src/lip_buddy_app.py",      # File for LipBuddyApp class
    "src/utils.py",              # File for utility functions like load_video, load_alignments
    ".env",                      # Environment file for storing sensitive data
    "setup.py",                  # Setup file for packaging
    "research/trials.ipynb",     # Jupyter notebook for experiments
    "tests/test_lip_reading_model.py",  # Test case for LipReadingModel
    "tests/test_video_processor.py",    # Test case for VideoProcessor
    "tests/test_lip_buddy_app.py",      # Test case for LipBuddyApp
    "tests/test_utils.py",              # Test case for utility functions
    "app.py",                    # Entry point for the application
    "store_index.py",            # Index file for storing database or cache-related logic
    "static/.gitkeep",           # Keeps the static folder in Git (for CSS, JS)
    "templates/chat.html"        # HTML template for the web interface
]

# Loop over the list of files and create the necessary directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directories if they don't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    # Create the file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass  # Create an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists and is not empty")
