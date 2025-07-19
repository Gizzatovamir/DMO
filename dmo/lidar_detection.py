import rclpy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import Odometry
import laser_geometry.laser_geometry as lg
from laser_geometry import LaserProjection
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from scipy.stats import kurtosis
from collections import deque
from sklearn.cluster import DBSCAN, MeanShift, AgglomerativeClustering, HDBSCAN
import message_filters
import tf_transformations as tr
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt
from matplotlib import animation
from .utils import Bbox
from visualization_msgs.msg import MarkerArray

EPS = 1e-9
STEP = 0.01
DIMENSIONS = 2
LASERSCAN_SIZE = 720
MAX_DISTANCE = 20

def add_prob(pcl: np.array, prob: np.array = None):
    if prob is not None:
        assert prob.shape == (LASERSCAN_SIZE,)
        if pcl.shape == (LASERSCAN_SIZE, 3):
            pcl[:, -1] = prob
            return 
    return np.hstack((pcl, np.zeros(((len(pcl), 1)))))
        


def get_prob_vector(distance_mat, rows, cols) -> np.array:
    res: list[float] = list()
    for i, j in zip(rows, cols):
        if distance_mat[i][j] < MAX_DISTANCE:
            res.append(distance_mat[i][j])
        else:
            res.append(0.5)
    return np.array(res)/sum(res)

def calc_probability(pcl_1: np.array, pcl_2: np.array):
    M = distance_matrix(pcl_1, pcl_2)
    M[~np.isfinite(M)] = 99999.
    rows, cols = linear_sum_assignment(M)
    return get_prob_vector(M, rows, cols), rows, cols

def reject_outliers(data, m=2):
    mean_diff = abs(data - np.mean(data, axis=0))
    var = m * np.std(data, axis=0)
    indx = np.all(mean_diff <= var, axis=1)
    return data[indx]

def get_t_from_odom(odom: Odometry) -> tuple[np.array, np.array]:
    T = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z])
    R = np.array(tr.quaternion_matrix([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]))
    return T[: DIMENSIONS], R[:DIMENSIONS, :DIMENSIONS].T

def apply_t(points: np.array, t: np.array, r: np.array):
    return points @ r.T - t

def get_medoid(vX):
  vMean = np.mean(vX, axis=0)                               # compute centroid
  return vX[np.argmin([sum((x - vMean)**2) for x in vX])]   # pick a point closest to centroid

def get_translate(odom_1: Odometry, odom_2: Odometry) -> tuple[np.array, np.array]:
    T = np.array([odom_1.pose.pose.position.x, odom_1.pose.pose.position.y]) - np.array([odom_2.pose.pose.position.x, odom_2.pose.pose.position.y])
    first_m = np.array(tr.quaternion_matrix([odom_1.pose.pose.orientation.x, odom_1.pose.pose.orientation.y, odom_1.pose.pose.orientation.z, odom_1.pose.pose.orientation.w]))[:2, :2]
    second_m = np.array(tr.quaternion_matrix([odom_2.pose.pose.orientation.x, odom_2.pose.pose.orientation.y, odom_2.pose.pose.orientation.z, odom_2.pose.pose.orientation.w]))[:2, :2]
    R = first_m @ second_m.T
    T = second_m.T @ T 
    return T, R

def pol2cart(rho, phi):
    res: np.array = np.empty((0,2))
    if rho < 12:
        x = rho* np.cos(phi)
        y = rho * np.sin(phi)
    else:
        x = 999
        y = 999
    res = np.vstack((res, (x,y)))
    return res


def laserscan_to_pcl(msg: LaserScan, old_msg: LaserScan) -> tuple[np.array, np.array, np.array]:
    cur_angle = msg.angle_min
    angle_inc = msg.angle_increment
    res: np.array = np.empty((0,DIMENSIONS))
    old_res: np.array = np.empty((0,DIMENSIONS))
    for index, (range, old_range) in enumerate(zip(msg.ranges, old_msg.ranges)):
        points = pol2cart(range, cur_angle)
        old_points = pol2cart(old_range, cur_angle)
        res = np.vstack((res, points))
        old_res = np.vstack((old_res, old_points))
        cur_angle += angle_inc
    return res, old_res

def single_laserscan_to_pcl(msg: LaserScan) -> np.array:
    res: np.array = np.empty((0,DIMENSIONS))
    cur_angle = msg.angle_min
    for index, range in enumerate(msg.ranges):
        points = pol2cart(range, cur_angle)
        res = np.vstack((res, points))
        cur_angle += msg.angle_increment

    return res

class DynamicObjectDetector(Node):
    def __init__(self):
        super().__init__("object_detector")
        self.laser_sub = message_filters.Subscriber(self, LaserScan, '/LidarFrontLeft/scan', qos_profile=qos_profile_sensor_data)
        # self.laser_sub = message_filters.Subscriber(self, LaserScan, '/merged/scan', qos_profile=qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(PoseArray, '/motion', 10)
        self.cluster_pub = self.create_publisher(LaserScan, '/clustered_scan', 10)
        self.cluster_pub_1 = self.create_publisher(LaserScan, '/clustered_scan_1', 10)
        self.point_publisher = self.create_publisher(PointCloud2, '/current_points', 10)
        self.point_publisher_1 = self.create_publisher(PointCloud2, '/last_points', 10)
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/odom', qos_profile=qos_profile_sensor_data)
        self.max_len: int = 6
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.laser_sub, self.odom_sub], queue_size=self.max_len, slop=0.01
        )
        self.synchronizer.registerCallback(self.sync_callback)

        self.min_upper_bound: float = 0.05
        self.min_bound: float = 2
        self.deque_index: int = -1
        self.i = 0

        # measure deque
        self.last_msgs: deque = deque(maxlen=self.max_len)
        self.last_scans: deque[LaserScan] = deque(maxlen=self.max_len)
        self.odom_queue: deque[Odometry] = deque(maxlen=self.max_len)
        self.bbox = Bbox(0.04)
        self.cur_marker_pub = self.create_publisher(MarkerArray, '/cur_markers', 10)
        self.last_marker_pub = self.create_publisher(MarkerArray, '/last_markers', 10)
    
    @property
    def header(self) -> Header:
        header = Header()
        header.frame_id = 'lidar_frontLeft_link'
        header.stamp = self.get_clock().now().to_msg()
        return header

    def get_clust_diff(self, new_points: list[np.array], old_points: list[np.array]):
        def get_mean_values(input_array: list[np.array]) -> tuple[np.array, list[int]]:
            res_mean: np.array = np.empty((0,DIMENSIONS))
            for cluster_points in input_array:
                # res_mean = np.vstack((res_mean, np.mean(cluster_points, axis=0)))
                res_mean = np.vstack((res_mean, get_medoid(cluster_points)))
            return res_mean

        new_mean = get_mean_values(new_points)
        old_mean = get_mean_values(old_points)
        cluster_dist_matrix = distance_matrix(new_mean, old_mean)
        rows, cols = linear_sum_assignment(cluster_dist_matrix)
        diff_list = list()
        for i, j in zip(rows, cols):
            diff_list.append(cluster_dist_matrix[i][j])
        return new_mean, np.hstack((np.array(diff_list), np.zeros(len(new_mean) - len(rows))))
               

    def get_clusters(self, points: np.array, l2_norm: np.array) -> tuple[np.array, dict[int, np.array]]:
        """
        Method to cluster laserscan points 
        """
        # cluster_data = np.hstack((np.nan_to_num(points, copy=False, nan=0.0, posinf=0.0, neginf=0.0), np.c_[angle], np.c_[np.nan_to_num(l2_norm, copy=False, nan=999.)]))
        # cluster_data = np.hstack((np.nan_to_num(points, copy=False, nan=999.0, posinf=999.0, neginf=0.0), np.c_[np.nan_to_num(l2_norm, copy=False, nan=999., posinf=999.)]))
        cluster_data = np.nan_to_num(points, copy=False, nan=999.0, posinf=999.0, neginf=0.0)
        # cluster_data = np.nan_to_num(points, copy=False, nan=0., posinf=0., neginf=0.0)
        self.cluster_model.fit(points[:, :2])
        clusters: list[np.array] = list()
        for index,lbl in enumerate(np.unique(self.cluster_model.labels_)):
            if lbl > 0:
                cluster_idx = self.cluster_model.labels_ == lbl
                cluster = points[cluster_idx]
                clusters.append(cluster)
        return clusters

    def get_diff(self, curr_scan: LaserScan, odom: Odometry) -> tuple[np.array, np.array, np.array]:
        curr_points, last_points = laserscan_to_pcl(curr_scan, self.last_msgs[self.deque_index])
        curr_t, curr_r = get_t_from_odom(odom)  
        last_t, last_r = get_t_from_odom(self.odom_queue[self.deque_index])  
        # self.get_logger().info(f't - {curr_t}, r - {curr_r}')
        # self.get_logger().info(f't - {last_t}, r - {last_r}')
        # self.get_logger().info(f'p - {curr_points}')
        # self.get_logger().info(f'l_p - {last_points}')
        last_points = np.matmul(last_points,last_r) + last_t
        curr_points = np.matmul(curr_points,curr_r) + curr_t
        curr_markers = self.bbox.run(curr_points, curr_scan.header)
        last_markers = self.bbox.run(last_points, curr_scan.header)

        self.cur_marker_pub.publish(curr_markers)
        self.last_marker_pub.publish(last_markers)

        curr_pcl = point_cloud2.create_cloud_xyz32(self.header, np.hstack((curr_points, np.zeros((len(curr_points), 1)))))
        last_pcl = point_cloud2.create_cloud_xyz32(self.header, np.hstack((last_points, np.zeros((len(last_points), 1)))))

        self.point_publisher.publish(curr_pcl)
        self.point_publisher_1.publish(last_pcl)

    def sync_callback(self, scan: LaserScan, odom: Odometry) -> None:
        # pcl = single_laserscan_to_pcl(scan)
        if len(self.last_msgs) >= self.max_len:
            self.get_diff(scan, odom)

        self.last_msgs.append(scan)
        self.odom_queue.append(odom)

def main():
    rclpy.init()
    rclpy.spin(DynamicObjectDetector())
    rclpy.shutdown()

if __name__ == "__main__":
    main()

