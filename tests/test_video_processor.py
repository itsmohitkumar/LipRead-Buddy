import unittest
from unittest.mock import patch, MagicMock
from video_processor import VideoProcessor

class TestVideoProcessor(unittest.TestCase):

    @patch('os.system')
    def test_convert_to_mp4(self, mock_system):
        """Test that video is converted to mp4 format."""
        video_processor = VideoProcessor('fake_path.mpg')
        video_processor.convert_to_mp4()

        # Ensure the system command to convert video is called
        mock_system.assert_called_once_with('ffmpeg -i fake_path.mpg -vcodec libx264 test_video.mp4 -y')

    @patch('cv2.VideoCapture')
    def test_load_video(self, mock_video_capture):
        """Test that video frames are loaded and processed correctly."""
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap

        # Mock the video capture and frame reading
        mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]  # First read is successful, second fails (end of video)

        video_processor = VideoProcessor('fake_path.mpg')
        frames = video_processor.load_video()

        # Ensure the video was captured and frames were processed
        mock_video_capture.assert_called_once_with('fake_path.mpg')
        mock_cap.read.assert_called()

    @patch('imageio.mimsave')
    def test_save_as_gif(self, mock_mimsave):
        """Test that video frames are saved as GIF."""
        video_processor = VideoProcessor('fake_path.mpg')

        # Fake video frames data
        frames = [MagicMock()]

        # Call the save_as_gif function
        video_processor.save_as_gif(frames)

        # Ensure the mimsave method is called
        mock_mimsave.assert_called_once_with('animation.gif', frames, fps=10)

if __name__ == '__main__':
    unittest.main()
