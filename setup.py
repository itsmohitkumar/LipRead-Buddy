from setuptools import setup, find_packages

# Read the README file for long description
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='lipbuddy',  # Your project name
    version='0.1.0',  # Project version
    author='Mohit Kumar',  # Your name
    author_email='mohitpanghal12345@gmail.com',  # Your email
    description='A lip-reading application using deep learning and Streamlit',  # Short description
    long_description=long_description,  # Long description from the README file
    long_description_content_type='text/markdown',  # README file format
    url='https://github.com/itsmohitkumar/LipRead-Buddy.git',  # Project URL, e.g., GitHub repository
    packages=find_packages(),  # Automatically find all packages
    include_package_data=True,  # Include additional files specified in MANIFEST.in
    install_requires=[
        'tensorflow>=2.0',
        'opencv-python',
        'imageio',
        'streamlit',
        'numpy',
        'ffmpeg-python',  # For video conversion
        'pytest',  # If you use pytest for testing
        'mock',  # For mocking in unit tests
        'pillow',  # For image manipulation
    ],  # List of dependencies
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],  # Additional metadata about the project
    python_requires='>=3.9',  # Minimum Python version required
    entry_points={
        'console_scripts': [
            'lipbuddy=app:main',  # Command to run your app, e.g., `lipbuddy`
        ],
    },
)
