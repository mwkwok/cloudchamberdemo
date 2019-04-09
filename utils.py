import cv2
import pandas as pd
import tensorflow as tf
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.visualization import draw_box, draw_caption

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
  
def convert_img(frame, tmp_file_path):
    cv2.imwrite(tmp_file_path, frame)
    return read_image_bgr(tmp_file_path)

def iou(box1, box2):
    xI1 = max(box1[0], box2[0])
    yI1 = max(box1[1], box2[1])
    xI2 = min(box1[2], box2[2])
    yI2 = min(box1[3], box2[3])

    aI  = max(0,(xI2 - xI1 + 1)) * max(0,(yI2 - yI1 + 1))
    aB1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    aB2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return aI / (aB1+aB2-aI)
  
def draw_label(frame, box, score, label, track, scale):
    draw_box(frame, (box/scale).astype(int), color=label_color(label) )
    caption = "{} {:.3f}".format(track, score)
    draw_caption(frame, (box/scale).astype(int), caption)
    return frame
      
class tracks_tracker():
    def __init__(self, time_scale):
        self.current_tracks = []
        self.all_tracks = []
        self.time_scale = time_scale
    
    def add_track(self, time, track, box):
        self._renew_current_tracks(time)
        
        track = {'time': time, 'track': track, 'box': box, 'update_time': time}
        
        if self._is_new_track(track, time):
            self.current_tracks.append(track)
            self.all_tracks.append(track)
            return True
          
        return False
      
    def get_tracks_record(self, time_step):
        record = pd.DataFrame(self.all_tracks)
        record['time_step'] = (record['time'] / time_step).astype(int) * time_step
        return record.groupby(['time_step','track']).size().unstack(fill_value=0)
        
    def _renew_current_tracks(self, time):
        self.current_tracks = [ct for ct in self.current_tracks if time - ct['update_time'] < self.time_scale]
    
    def _is_new_track(self, track, time):
        is_new_track = True
        for current_track in self.current_tracks:
            # if the new track overlaps with current track, reset current tack's update_time
            if iou( track['box'], current_track['box'] ) >= 0.5:
                current_track['update_time'] = time
                is_new_track = False
        return is_new_track
