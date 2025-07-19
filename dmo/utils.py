import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header

class Point2D:
    def __init__(self, x:float=0, y:float=0):
        self.x:float = x
        self.y:float = y

class Rect:
    def __init__(self, points: np.array):
        self.points: np.array = points
        self.p0, self.p1, self.p2, self.p3 = self.get_sides(self.points)
    
    @property
    def edges(self) -> tuple[Point2D, Point2D, Point2D, Point2D]:
        return self.p0, self.p1, self.p2, self.p3
        

    @staticmethod
    def get_sides(points: np.array) -> tuple[Point2D, Point2D, Point2D, Point2D]:
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        return Point2D(min_x, min_y), Point2D(max_x, min_y), Point2D(max_x, max_y), Point2D(min_x, max_y)

class Bbox(object):
    def __init__(self, padding: float):
        self.padding: float = padding # padding from edge point to bbox side
        self.model = DBSCAN(eps=0.1, min_samples=10)
        
    
    def get_bbox(self, points: np.array) -> tuple[Rect]:
        self.model.fit(points)
        rects: list[Rect] = list()
        for label in np.unique(self.model.labels_):
            if label >= 0:
                cluster_idx = self.model.labels_ == label
                cluster = points[cluster_idx]
                rects.append(Rect(cluster))
        return rects

    def run(self, points:np.array, header: Header) -> MarkerArray:
        rects = self.get_bbox(points)
        markers = MarkerArray()
        for index, rect in enumerate(rects):
            marker = Marker()
            marker.header = header
            marker.id = index
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.1  # Line width
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            m_points = []
            for edge in rect.edges:
                point = Point()
                point.x = edge.x
                point.y = edge.y
                point.z = 0.0
                m_points.append(point)
            # print(m_points)
            marker.points = m_points
            markers.markers.append(marker)
        return markers
    
if __name__ == '__main__':
    pcl_1 = np.random.normal(3, 0.05, size=(720,2))
    pcl_2 = np.random.normal(6, 0.1, size=(720,2))
    pcl_3 = np.random.normal(10, 2.5, size=(720,2))
    plt.plot([el[0] for el in pcl_1],[el[1] for el in pcl_1], 'r.')
    plt.plot([el[0] for el in pcl_2],[el[1] for el in pcl_2], 'b.')
    plt.plot([el[0] for el in pcl_3],[el[1] for el in pcl_3], 'g.')
    pcl = np.vstack((pcl_1, pcl_2, pcl_3))
    bbox = Bbox(0.01)
    rects = bbox.get_bbox(pcl)
    print(rects)
    print(pcl)
    for rect in rects:
        points = rect.edges
        print(points)
        for el, el_1 in zip(points[:-1], points[1:]):
            plt.plot([el.x, el_1.x], [el.y, el_1.y])
        plt.plot([points[0].x, points[-1].x], [points[0].y, points[-1].y])
    plt.show()

