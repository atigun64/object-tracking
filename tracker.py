import numpy as np
from geometry import distance
from models import Track, Detection
from config import DIST_THRESHOLD, MISS_THRESHOLD, CONFIDENCE_THRESHOLD

from matching import Matrix, hungarian_match

def det_distance(det1, track1):
    dist_box = distance(det1.box, track1.box)
    dist_category = 0 if det1.name_id == track1.name_id else 1
    return dist_box + dist_category * 10  # Adjust the weight for category distance as needed

class Tracker:
    def __init__(self):
        self.cur_id = 0
        self.tracks = []
    
    def update_demo(self, detections):
        distance_matrix = Matrix(len(detections), len(self.tracks))

        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                dist = det_distance(det, track)
                if dist < DIST_THRESHOLD and det.conf > CONFIDENCE_THRESHOLD:
                    distance_matrix.set(i, j, dist)
                else:
                    distance_matrix.set(i, j, float("inf"))
        
        matchings = hungarian_match(distance_matrix, maximize=False)
        
        matched_old = set()
        matched_new = set()

        updated_tracks = []

        for i, j in matchings:
            updated_tracks.append(Track(
                box=np.array(detections[i].box),
                name=detections[i].name,
                conf=detections[i].conf,
                id=self.tracks[j].id,
                name_id=self.tracks[j].name_id
            ))
            matched_new.add(i)
            matched_old.add(j)

        for i in range(len(detections)):
            if i not in matched_new:
                print("New object: ", detections[i])
                updated_tracks.append(Track(
                    box=np.array(detections[i].box),
                    name=detections[i].name,
                    conf=detections[i].conf,
                    id=self.cur_id,
                    name_id=detections[i].name_id
                ))
                self.cur_id += 1

        for j in range(len(self.tracks)):
            if j not in matched_old:
                if self.tracks[j].missed + 1 > MISS_THRESHOLD:
                    print("Object disappeared: ", (self.tracks[j].box, self.tracks[j].name, self.tracks[j].conf))
                else:
                    updated_tracks.append(Track(
                        box=np.array(self.tracks[j].box),
                        name=self.tracks[j].name,
                        conf=self.tracks[j].conf,
                        id=self.tracks[j].id,
                        missed=self.tracks[j].missed + 1,
                        name_id=self.tracks[j].name_id
                    ))

        self.tracks = updated_tracks

    def update(self, detections):
        matched = set()
        updated_tracks = []

        for det in detections:
            best_dist = float("inf")
            best_track = None
            for track in self.tracks:
                if track.id in matched:
                    continue
                dist = det_distance(det, track)
                if dist < best_dist:
                    best_dist = dist
                    best_track = track

            if best_track and best_dist < DIST_THRESHOLD and det.conf > CONFIDENCE_THRESHOLD:
                updated_tracks.append(Track(
                    box=np.array(det.box),
                    name=det.name,
                    conf=det.conf,
                    id=best_track.id,
                    name_id=best_track.name_id
                ))
                matched.add(best_track.id)
            elif det.conf > CONFIDENCE_THRESHOLD:
                print("New object: ", det)
                updated_tracks.append(Track(
                    box=np.array(det.box),
                    name=det.name,
                    conf=det.conf,
                    id=self.cur_id,
                    name_id=det.name_id
                ))
                self.cur_id += 1

        for track in self.tracks:
            if track.id not in matched:
                if track.missed + 1 > MISS_THRESHOLD:
                    print("Object disappeared: ", (track.box, track.name, track.conf))
                else:
                    updated_tracks.append(Track(
                        box=np.array(track.box),
                        name=track.name,
                        conf=track.conf,
                        id=track.id,
                        missed=track.missed + 1,
                        name_id=track.name_id
                    ))

        self.tracks = updated_tracks