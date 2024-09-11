import os
import cv2
import tensorflow as tf
import imageio
from logger import logger

class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path

    def convert_to_mp4(self, output_name='test_video.mp4'):
        """Converts the input video to mp4 format using ffmpeg."""
        logger.info(f"Converting {self.video_path} to mp4 format...")
        try:
            os.system(f'ffmpeg -i {self.video_path} -vcodec libx264 {output_name} -y')
            logger.info("Video converted successfully.")
        except Exception as e:
            logger.error(f"Error converting video: {e}")
            raise

    def load_video(self) -> tf.Tensor:
        """Loads video frames and converts them to grayscale."""
        logger.info(f"Loading video from {self.video_path}...")
        try:
            cap = cv2.VideoCapture(self.video_path)
            frames = []
            for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = tf.image.rgb_to_grayscale(frame)
                frames.append(frame[190:236, 80:220, :])  # Crop frame
            cap.release()

            mean = tf.math.reduce_mean(frames)
            std = tf.math.reduce_std(tf.cast(frames, tf.float32))
            logger.info("Video loaded and processed successfully.")
            return tf.cast((frames - mean), tf.float32) / std
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            raise

    def save_as_gif(self, frames, gif_name='animation.gif', fps=10):
        """Saves video frames as a GIF."""
        logger.info(f"Saving video frames as {gif_name}...")
        try:
            imageio.mimsave(gif_name, frames, fps=fps)
            logger.info(f"GIF saved successfully as {gif_name}.")
        except Exception as e:
            logger.error(f"Error saving GIF: {e}")
            raise
