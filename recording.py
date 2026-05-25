import cv2
import threading
import time
import os
from datetime import datetime

class RTSPStreamer:
    def __init__(self, stream_name, rtsp_url):
        self.stream_name = stream_name
        
        # GStreamer pipeline string optimized for low latency
        self.gst_pipeline = (
            f"rtspsrc location={rtsp_url} latency=0 ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
        )
        
        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open stream {self.stream_name}")
            
        self.frame = None
        self.ret = False
        self.running = True
        
        # Recording State
        self.is_recording = False
        self.video_writer = None
        
        # Start background thread to read frames
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        """Continuously pulls frames from the GStreamer appsink."""
        while self.running:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            else:
                time.sleep(0.1)

    def toggle_recording(self):
        """Starts or stops recording the current stream."""
        if not self.is_recording:
            if self.frame is not None:
                # Initialize VideoWriter
                height, width = self.frame.shape[:2]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.stream_name}_record_{timestamp}.mp4"
                
                # Use mp4v codec for standard MP4 output
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
                self.is_recording = True
                print(f"[{self.stream_name}] Started recording: {filename}")
        else:
            # Stop Recording
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print(f"[{self.stream_name}] Stopped recording.")

    def take_screenshot(self):
        """Saves the current frame to a JPG."""
        if self.frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{self.stream_name}_shot_{timestamp}.jpg"
            cv2.imwrite(filename, self.frame)
            print(f"[{self.stream_name}] Screenshot saved: {filename}")

    def process_frame(self):
        """Handles frame writing if recording is active, and returns the frame for display."""
        if self.frame is not None:
            frame_copy = self.frame.copy()
            
            # If recording, write the clean frame to the video file
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame_copy)
                
            # Add visual indicator for recording on the display frame
            if self.is_recording:
                cv2.circle(frame_copy, (30, 30), 10, (0, 0, 255), -1) # Red dot
                cv2.putText(frame_copy, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            return frame_copy
        return None

    def release(self):
        self.running = False
        self.thread.join()
        if self.is_recording and self.video_writer:
            self.video_writer.release()
        self.cap.release()


def main():
    # Replace these with your actual RTSP URLs
    rtsp_url_1 = "rtsp://root:root@192.168.6.91/cam1/h264"
    rtsp_url_2 = "rtsp://root:root@192.168.6.90/cam1/h264"

    print("Initializing Streams...")
    stream1 = RTSPStreamer("Cam_1", rtsp_url_1)
    stream2 = RTSPStreamer("Cam_2", rtsp_url_2)

    print("\n--- Controls ---")
    print("Press '1' to toggle recording for Camera 1")
    print("Press '2' to toggle recording for Camera 2")
    print("Press 'c' to take screenshots of BOTH cameras")
    print("Press 'q' to Quit\n")

    while True:
        frame1 = stream1.process_frame()
        frame2 = stream2.process_frame()

        if frame1 is not None:
            cv2.imshow("Camera 1", frame1)
        if frame2 is not None:
            cv2.imshow("Camera 2", frame2)

        # Keyboard Input Handling
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('1'):
            stream1.toggle_recording()
        elif key == ord('2'):
            stream2.toggle_recording()
        elif key == ord('c'):
            stream1.take_screenshot()
            stream2.take_screenshot()

    # Cleanup
    stream1.release()
    stream2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()