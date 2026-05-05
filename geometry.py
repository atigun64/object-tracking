def distancePoint(pt0, pt1):
    dx = pt0[0] - pt1[0]
    dy = pt0[1] - pt1[1]
    return dx*dx + dy*dy

def distance(box0, box1):
    return max(
        distancePoint((box0[0], box0[1]), (box1[0], box1[1])),
        distancePoint((box0[2], box0[3]), (box1[2], box1[3])),
        distancePoint((box0[0], box0[3]), (box1[0], box1[3])),
        distancePoint((box0[2], box0[1]), (box1[2], box1[1]))
    )
    
class Geometry:
    def __init__(self):
        pass
    
    def midpoint(self, box):
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    def distance(self, box0, box1):
        (x0, y0) = self.midpoint(box0)
        (x1, y1) = self.midpoint(box1)
        return (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1)