import unittest
from unittest.mock import patch, MagicMock
from model import LipReadingModel

class TestLipReadingModel(unittest.TestCase):

    @patch('tensorflow.keras.models.Sequential.load_weights')
    def test_model_build_and_load_weights(self, mock_load_weights):
        """Test that the model is built and weights are loaded correctly."""
        mock_load_weights.return_value = None  # Mock the load weights method

        # Initialize the model
        model = LipReadingModel()

        # Check that the model has the expected number of layers
        self.assertEqual(len(model.model.layers), 12)

        # Assert that the weights loading function was called
        mock_load_weights.assert_called_once()

    @patch('tensorflow.keras.models.Sequential.predict')
    def test_model_predict(self, mock_predict):
        """Test model prediction function."""
        mock_predict.return_value = MagicMock()  # Mock the prediction result
        model = LipReadingModel()

        # Fake input video data
        video_data = MagicMock()

        # Call the predict method
        model.predict(video_data)

        # Check that the predict method was called
        mock_predict.assert_called_once_with(video_data)

if __name__ == '__main__':
    unittest.main()
