import cv2
import time
from concurrent.futures import ThreadPoolExecutor


class VideoProcessor:
    def __init__(self, camera_inputs):
        # initialize the VideoProcessor with multiple camera inputs in the alone way (synchronization)
        self.cameras = [cv2.VideoCapture(input) for input in camera_inputs] # 1 or 3 or another number of camera gathering
        self.writers = [None] * len(camera_inputs)
        self.start_time = time.time()
        self.frame_buffer = []
        self.buffer_size = 30  # adjust based on available memory

    def get_frames(self):
        # generator that yields frames from ALL cameras
        with ThreadPoolExecutor(max_workers=len(self.cameras)) as executor:
            while True:
                futures = [executor.submit(self._read_frame, i) for i in range(len(self.cameras))]
                for future in futures:
                    frame, camera_id = future.result()
                    if frame is not None:
                        yield frame, camera_id
                if time.time() - self.start_time > 960:  # at most 16 minutes
                    break

    def _read_frame(self, camera_id):
        # read a frame from a specific camera
        ret, frame = self.cameras[camera_id].read()
        if ret:
            return frame, camera_id
        return None, camera_id

    def write_frame(self, frame, camera_id):
        # write a frame to the output video file for a specific camera
        if self.writers[camera_id] is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writers[camera_id] = cv2.VideoWriter(f'output_camera_{camera_id}.mp4', fourcc, 30.0, (frame.shape[1], frame.shape[0]))
        self.writers[camera_id].write(frame)
        self.frame_buffer.append((frame, camera_id))
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)


    def display_frame(self, frame):
        # display a frame in a window
        cv2.imshow('Live Race', frame)
        cv2.waitKey(1)

    def get_output_videos(self):
        # get the paths of all output video files
        return [writer.getOutputFileName() for writer in self.writers if writer is not None]

    @property
    def elapsed_time(self):
        # get the elapsed time since the VideoProcessor was initialized
        return time.time() - self.start_time

    def get_recent_frames(self, num_frames=10):
        # get the most recent frames from the frame buffer.
        return self.frame_buffer[-num_frames:]

    def __del__(self):
        # destructor to release all resources.
        for cap in self.cameras:
            cap.release()
        for writer in self.writers:
            if writer is not None:
                writer.release()
        cv2.destroyAllWindows()
