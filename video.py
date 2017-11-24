"""
Video Player Module.

@author: Gary Corcoran
@date: Nov. 24th, 2017
"""

import cv2

class Video():
    """ Video player. """
    def __init__(self, video_path=None):
        """
        Initialize parameters.

        @param  video_path: path to input video file
        """
        self.video_path = video_path
        self.__cap = None

    def set_video_path(self, video_path):
        """
        Set path to input video file.

        @param  video_path: path to input video file

        @modifies   self.video_path: stores video file
        """
        self.video_path = video_path

    def get_video_path(self):
        """
        Returns video path.

        @return video_path: path to input video file
        """
        return self.video_path

    def run(self):
        """
        Play video stored in video_path.
        """
        if self.video_path is None:
            print('Please input video path before running')
            return
        self.__cap = cv2.VideoCapture(0)
        print(self.__cap.isOpened())
        while self.__cap.isOpened():
            # read frame
            ret, frame = self.__cap.read()
            print(ret)
            # display
            cv2.imshow('Frame', frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        self.__cap.release()
        cv2.destroyAllWindows()
        

def main():
    """ Main Function. """
    video_path = 'dataset/IMG_0687.MOV'
    video_path = '/home/gary/opencv-3.2.0/samples/data/vtest.avi'

    # create video player object
    vod = Video(video_path)
    vod.run()

if __name__ == '__main__':
    main()
