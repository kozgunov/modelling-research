import cv2
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class VideoProcessor:
    def __init__(self, camera_inputs):
        """
        Initialize the VideoProcessor with multiple camera inputs.
        
        Args:
        camera_inputs (list): List of paths or camera indices for video inputs.
        """
        self.cameras = [cv2.VideoCapture(input) for input in camera_inputs]
        self.writers = [None] * len(camera_inputs)
        self.start_time = time.time()
        self.frame_buffer = []
        self.buffer_size = 30  # Adjust based on available memory

    def get_frames(self):
        """
        Generator that yields frames from all cameras.
        
        Yields:
        tuple: (frame, camera_id)
        """
        with ThreadPoolExecutor(max_workers=len(self.cameras)) as executor:
            while True:
                futures = [executor.submit(self._read_frame, i) for i in range(len(self.cameras))]
                for future in futures:
                    frame, camera_id = future.result()
                    if frame is not None:
                        yield frame, camera_id
                if time.time() - self.start_time > 900:  # 15 minutes
                    break

    def _read_frame(self, camera_id):
        """
        Read a frame from a specific camera.
        
        Args:
        camera_id (int): Index of the camera to read from.
        
        Returns:
        tuple: (frame, camera_id) or (None, camera_id) if no frame is available.
        """
        ret, frame = self.cameras[camera_id].read()
        if ret:
            return frame, camera_id
        return None, camera_id

    def write_frame(self, frame, camera_id):
        """
        Write a frame to the output video file for a specific camera.
        
        Args:
        frame (np.array): The frame to write.
        camera_id (int): The ID of the camera that captured this frame.
        """
        if self.writers[camera_id] is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writers[camera_id] = cv2.VideoWriter(f'output_camera_{camera_id}.mp4', fourcc, 30.0, (frame.shape[1], frame.shape[0]))
        self.writers[camera_id].write(frame)
        self.frame_buffer.append((frame, camera_id))
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

    def display_frame(self, frame):
        """
        Display a frame in a window.
        
        Args:
        frame (np.array): The frame to display.
        """
        cv2.imshow('Live Race', frame)
        cv2.waitKey(1)

    def get_output_videos(self):
        """
        Get the paths of all output video files.
        
        Returns:
        list: List of output video file paths.
        """
        return [writer.getOutputFileName() for writer in self.writers if writer is not None]

    @property
    def elapsed_time(self):
        """
        Get the elapsed time since the VideoProcessor was initialized.
        
        Returns:
        float: Elapsed time in seconds.
        """
        return time.time() - self.start_time

    def get_recent_frames(self, num_frames=10):
        """
        Get the most recent frames from the frame buffer.
        
        Args:
        num_frames (int): Number of recent frames to retrieve.
        
        Returns:
        list: List of recent (frame, camera_id) tuples.
        """
        return self.frame_buffer[-num_frames:]

    def __del__(self):
        """
        Destructor to release all resources.
        """
        for cap in self.cameras:
            cap.release()
        for writer in self.writers:
            if writer is not None:
                writer.release()
        cv2.destroyAllWindows()
