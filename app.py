import streamlit as st
import os
from video_processor import VideoProcessor
from model import LipReadingModel
from utils import load_alignments, num_to_char
from logger import logger

class LipBuddyApp:
    def __init__(self):
        self.model = LipReadingModel()

    def setup_sidebar(self):
        """Sets up the sidebar for the Streamlit app."""
        with st.sidebar:
            st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
            st.title('LipBuddy')
            st.info('This application is originally developed from the LipNet deep learning model.')

    def display_video(self, selected_video: str, col):
        """Displays the selected video in the provided Streamlit column."""
        try:
            file_path = os.path.join('..', 'data', 's1', selected_video)
            video_processor = VideoProcessor(file_path)
            video_processor.convert_to_mp4()
            with open('test_video.mp4', 'rb') as video_file:
                video_bytes = video_file.read()
                col.video(video_bytes)
        except Exception as e:
            logger.error(f"Error displaying video: {e}")
            st.error(f"Error displaying video: {e}")

    def render_model_output(self, selected_video: str, col):
        """Processes and displays the model output in the provided column."""
        try:
            video_file_path = os.path.join('..', 'data', 's1', selected_video)
            video_processor = VideoProcessor(video_file_path)
            video, annotations = self.load_data(video_file_path)

            # Display video frames as GIF
            video_processor.save_as_gif(video)
            col.image('animation.gif', width=400)

            # Make prediction
            prediction = self.model.predict(video)
            decoder = tf.keras.backend.ctc_decode(prediction, [75], greedy=True)[0][0].numpy()
            col.text(decoder)

            # Decode prediction to text
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            col.text(converted_prediction)
        except Exception as e:
            logger.error(f"Error rendering model output: {e}")
            st.error(f"Error rendering model output: {e}")

    def load_data(self, path: str):
        """Loads video frames and alignment tokens for the selected video."""
        try:
            video = VideoProcessor(path).load_video()
            alignment_path = os.path.join('..', 'data', 'alignments', 's1', os.path.splitext(os.path.basename(path))[0] + '.align')
            alignments = load_alignments(alignment_path)
            return video, alignments
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def run(self):
        """Runs the Streamlit app."""
        st.set_page_config(layout='wide')
        self.setup_sidebar()

        st.title('LipBuddy Lip Reading App')

        options = os.listdir(os.path.join('..', 'data', 's1'))
        selected_video = st.selectbox('Choose video', options)

        col1, col2 = st.columns(2)

        if selected_video:
            self.display_video(selected_video, col1)
            self.render_model_output(selected_video, col2)

if __name__ == "__main__":
    app = LipBuddyApp()
    app.run()
