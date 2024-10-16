import numpy as np
from scipy.optimize import linear_sum_assignment


class ShipTracker:
    def __init__(self, max_age=1, min_hits=3): # for live-tracking the boats
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.tracked_objects = None

    def _iou(self, bbox1, bbox2):
        #  computee IOU between two bounding boxes
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        iou = inter_area / (area1 + area2 - inter_area + 1e-6)
        return iou

    def handle_occlusions(self, tracked_objects):
        for i, obj1 in enumerate(tracked_objects):
            for j, obj2 in enumerate(tracked_objects[i+1:]):
                if self._iou(obj1['bbox'], obj2['bbox']) > 0.5:  # change threshold as needed
                    # handle occlusion
                    obj1['occluded'] = True
                    obj2['occluded'] = True
        return tracked_objects

    def update(self, detections):
        self.frame_count += 1
        self.tracked_objects = None
        
        if len(self.trackers) == 0:
            for det in detections:
                self.trackers.append({
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'hits': 1,
                    'age': 0,
                    'color': None
                })
        else:
            # simple prediction of new locations (closest some metres following from logic) 
            for t in self.trackers:
                t['age'] += 1
            
            iou_matrix = np.zeros((len(detections), len(self.trackers))) # match detections to trackers
            for d, det in enumerate(detections):
                for t, trk in enumerate(self.trackers):
                    iou_matrix[d, t] = self._iou(det['bbox'], trk['bbox'])
            
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.asarray(matched_indices).T
            
            unmatched_detections = []
            for d, det in enumerate(detections):
                if d not in matched_indices[:, 0]:
                    unmatched_detections.append(d)
            
            unmatched_trackers = []
            for t, trk in enumerate(self.trackers):
                if t not in matched_indices[:, 1]:
                    unmatched_trackers.append(t)
            
            # Update matched trackers
            for m in matched_indices:
                self.trackers[m[1]]['bbox'] = detections[m[0]]['bbox']
                self.trackers[m[1]]['hits'] += 1
                self.trackers[m[1]]['age'] = 0
            
            # Add new trackers
            for i in unmatched_detections:
                self.trackers.append({
                    'bbox': detections[i]['bbox'],
                    'class': detections[i]['class'],
                    'hits': 1,
                    'age': 0,
                    'color': None
                })
            
            # Remove dead trackers
            self.trackers = [t for t in self.trackers if t['age'] <= self.max_age and t['hits'] >= self.min_hits]
            
            self.tracked_objects = self.handle_occlusions(self.trackers)
        
        return self.tracked_objects

    def update_colors(self, colors):
        for t, color in zip(self.trackers, colors):
            if t['color'] is None:
                t['color'] = color
