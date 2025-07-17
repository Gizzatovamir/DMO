import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class Point:
    def __init__(self, x:float=0, y:float=0):
        self.x:float = x
        self.y:float = y

class Rect:
    def __init__(self, points: np.array):
        self.points: np.array = points
        self.p0, self.p1, self.p2, self.p3 = self.get_sides(self.points)
    
    @property
    def edges(self) -> tuple[Point, Point, Point, Point]:
        return self.p0, self.p1, self.p2, self.p3
        

    @staticmethod
    def get_sides(points: np.array) -> tuple[Point, Point, Point, Point]:
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        return Point(min_x, min_y), Point(max_x, min_y), Point(max_x, max_y), Point(min_x, max_y)

class Bbox(object):
    def __init__(self, padding: float):
        self.padding: float = padding # padding from edge point to bbox side
        self.model = DBSCAN(eps=0.05, min_samples=5)
        
    
    def get_bbox(self, points: np.array) -> tuple[Rect]:
        self.model.fit(points)
        rects: list[Rect] = list()
        for label in np.unique(self.model.labels_):
            if label > 0:
                cluster_idx = self.cluster_model.labels_ == label
                cluster = points[cluster_idx]
                rects.append(Rect(cluster))
        return rects