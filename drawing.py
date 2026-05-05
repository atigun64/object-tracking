import cv2
import numpy as np

from models import Track

def color_for_id(track_id: int) -> tuple[int, int, int]:
    np.random.seed(track_id)
    color = np.random.randint(80, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    draw_lost: bool = False,
) -> np.ndarray:
    output = frame.copy()

    for track in tracks:
        if track.missed > 0 and not draw_lost:
            continue

        x1, y1, x2, y2 = track.box.astype(int)
        color = color_for_id(track.id)

        label = f"#{track.id} {track.name} {track.conf:.2f}"

        if track.missed > 0:
            label += f" missed={track.missed}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return output
