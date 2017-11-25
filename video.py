"""
Video Player Module.

@author: Gary Corcoran
@date_created: Nov. 24th, 2017
"""

import numpy as np
import cv2

class Video():
    """ Video player. """
    def __init__(self, video_path=None, processor=None):
        """
        Initialize parameters.

        @param  video_path: path to input video file
        @param  processor:  frame processor object
        """
        self.video_path = video_path
        self._processor = processor
        self._cap = None

    def set_video_path(self, video_path):
        """
        Set path to input video file.

        @param  video_path: path to input video file

        @modifies   self.video_path:    stores video file
        """
        self.video_path = video_path

    def set_processor(self, processor):
        """
        Set frame processor.

        @param  processor:  frame processor object
        
        @modifies   self._processor:    stores object
        """
        self._processor = processor

    def get_video_path(self):
        """
        Returns video path.

        @return video_path: path to input video file
        """
        return self.video_path

    def _is_opened(self):
        """
        Check if video is opened.

        @return ret:    true if video is opened, else false
        """
        ret = self._cap.isOpened()
        if ret is False:
            print('Could not opened video file stored at:', self.video_path)
        return ret

    def _read(self):
        """
        Read video frame.

        @return ret:    false end of file, else true
        @return frame:  video frame
        """
        ret, frame = self._cap.read()
        if ret is False:
            print('Reached end of file.')
        return ret, frame

    def _release(self):
        """
        Release resources.
        """
        self._cap.release()
        cv2.destroyAllWindows()

    def _display(self, frame):
        """
        Display video frame.
        
        @return ret:    false if user quit video, else true
        """
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        # if user press 'q'
        if key == 113:
            print('User quit video.')
            return False
        return True

    def _rotate_crop(self, frame):
        """
        Rotate and crop frame accordingly.

        @param  frame:  input video frame

        @return frame:  resultant frame
        """
        # get new image dimensions
        rows, cols = frame.shape[:2]
        # rotate frame
        M = cv2.getRotationMatrix2D((cols/2, rows/2), -90, 1)
        # bottem left pixel
        bl = [0, rows, 1]
        # find new position
        bl = np.int_(np.dot(M, bl))
        frame = cv2.warpAffine(frame, M, (cols, rows))
        # crop frame (i.e. remove black columns) using transformed 
        # pixel location
        frame = frame[:, bl[0]:bl[1]]
        return frame

    def _process(self, frame):
        """
        Process video frame.

        @param  frame:  input video frame

        @return frame:  processed video frame
        @return feat:   feature space of frame if processor is set, else false
        """
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5) 
        frame = self._rotate_crop(frame)
        if self._processor is not None:
            self._processor.set_image(frame)
            feat = self._processor.compute()
            frame = self._processor.visualize()    
            return frame, feat
        return frame, False

    def run(self):
        """
        Play video stored in video_path.
        """
        print("Press 'q' to quit video.")
        if self.video_path is None:
            print('Please input video path before running')
            return
        self._cap = cv2.VideoCapture(self.video_path)
        while self._is_opened():
            # read frame
            ret, frame = self._cap.read()
            if ret is False:
                break
            #TODO PROCESS VIDEO FRAME
            frame, feat = self._process(frame)
            # display
            if self._display(frame) is False:
                break
        # release resources
        self._release() 

def main():
    """ Main Function. """
    import sys
    from hog import HOG

    # check command line inputs
    if len(sys.argv) >= 2:
        print(sys.argv[1])
        # set command line input to video path
        video_path = sys.argv[1]
    else:
        # set default video path
        video_path = 'dataset/IMG_0687.MOV'
    # HoG parameters
    hog_params = {'winSize': (384, 480), 'blockSize': (32, 32),
        'blockStride': (16, 16), 'cellSize': (16, 16), 'nbins': 9}    
    # create frame processor object
    hog = HOG(**hog_params)
    # create video player object
    vod = Video(video_path=video_path, processor=hog)
    vod.run()

if __name__ == '__main__':
    main()
