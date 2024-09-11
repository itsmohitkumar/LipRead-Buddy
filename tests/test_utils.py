import unittest
from unittest.mock import patch, mock_open
from utils import load_alignments, char_to_num

class TestUtils(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data="0 1000 a\n1000 2000 b\n2000 3000 sil\n")
    def test_load_alignments(self, mock_file):
        """Test that alignments are loaded correctly from a file."""
        result = load_alignments('fake_align_path.align')

        # Expected alignment tokens excluding 'sil'
        expected_tokens = char_to_num([b' ', b'a', b' ', b'b'])

        # Assert the expected tokens match the result
        self.assertTrue((result.numpy() == expected_tokens.numpy()).all())

if __name__ == '__main__':
    unittest.main()
