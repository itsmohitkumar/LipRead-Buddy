import unittest
from unittest.mock import patch, MagicMock
from app import LipBuddyApp

class TestLipBuddyApp(unittest.TestCase):

    @patch('streamlit.video')
    @patch('streamlit.sidebar.image')
    def test_display_video(self, mock_image, mock_video):
        """Test that the video is displayed in the Streamlit app."""
        app = LipBuddyApp()

        with patch('video_processor.VideoProcessor.convert_to_mp4', return_value=None), \
             patch('builtins.open', mock_open(read_data=b'test_video')):
            app.display_video('fake_video.mpg', MagicMock())

        mock_video.assert_called()

    @patch('streamlit.image')
    @patch('streamlit.text')
    def test_render_model_output(self, mock_text, mock_image):
        """Test that the model output is rendered in the Streamlit app."""
        app = LipBuddyApp()

        with patch('video_processor.VideoProcessor.load_video', return_value=MagicMock()), \
             patch('model.LipReadingModel.predict', return_value=MagicMock()), \
             patch('tensorflow.keras.backend.ctc_decode', return_value=[MagicMock()]):
            app.render_model_output('fake_video.mpg', MagicMock())

        mock_image.assert_called()
        mock_text.assert_called()

if __name__ == '__main__':
    unittest.main()
