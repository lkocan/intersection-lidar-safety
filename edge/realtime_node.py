import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import torch
import time
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.utils.preprocess import PointCloudPreprocessor, PillarConfig
from core.utils.inference import decode_predictions
from core.models.pointpillars import PointPillars
from core.tracking.tracker_3d import Tracker3D 

class LiDARRealtimeNode(Node):
    def __init__(self):
        super().__init__('lidar_realtime_node')
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Štartujem Edge Systém na zariadení: {self.device.upper()}")

        self.cfg = PillarConfig()
        self.preprocessor = PointCloudPreprocessor(cfg=self.cfg, device=self.device)

        self.get_logger().info("Načítavam PointPillars model...")
        self.model = PointPillars(self.cfg).to(self.device)
        # TODO: načítanie váh (.pth súbor)
        # self.model.load_state_dict(torch.load('core/models/weights/moje_vahy.pth'))
        self.model.eval()

        self.tracker = Tracker3D()

        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.lidar_callback,
            10
        )
        self.marker_pub = self.create_publisher(MarkerArray, '/livox/detections', 10)
        
        self.get_logger().info("Systém je pripravený. Čakám na dáta z LiDARu...")

    def publish_markers(self, tracks, header):
        marker_array = MarkerArray()
        
        if len(tracks) == 0:
            clear_marker = Marker()
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)
            self.marker_pub.publish(marker_array)
            return

        for track in tracks:
            x, y, z, w, l, h, yaw = track[0:7]
            class_id = int(track[7])
            track_id = int(track[8])

            box = Marker()
            box.header = header  
            box.ns = "boxes"
            box.id = track_id
            box.type = Marker.CUBE
            box.action = Marker.ADD

            box.pose.position.x = float(x)
            box.pose.position.y = float(y)
            box.pose.position.z = float(z)

            box.pose.orientation.x = 0.0
            box.pose.orientation.y = 0.0
            box.pose.orientation.z = math.sin(yaw / 2.0)
            box.pose.orientation.w = math.cos(yaw / 2.0)

            box.scale.x = float(l)  
            box.scale.y = float(w)  
            box.scale.z = float(h)  

            box.color.a = 0.4  
            if class_id == 0:
                box.color.r, box.color.g, box.color.b = 0.0, 1.0, 0.0
            else:
                box.color.r, box.color.g, box.color.b = 0.0, 0.5, 1.0

            box.lifetime.sec = 0
            box.lifetime.nanosec = 200000000
            marker_array.markers.append(box)

            text = Marker()
            text.header = header
            text.ns = "ids"
            text.id = track_id + 10000 
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            text.pose.position.x = float(x)
            text.pose.position.y = float(y)
            text.pose.position.z = float(z) + float(h) / 2.0 + 0.5 
            
            text.scale.z = 0.7  
            text.color.a, text.color.r, text.color.g, text.color.b = 1.0, 1.0, 1.0, 1.0
            text.text = f"ID: {track_id}"
            
            text.lifetime.sec = 0
            text.lifetime.nanosec = 200000000
            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)


    def lidar_callback(self, msg):
        start_time = time.perf_counter()

        gen = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        points = np.array(list(gen), dtype=np.float32)

        if len(points) == 0:
            return

        prep_data = self.preprocessor(points)
        if prep_data['pillars'].shape[1] == 0:
            return

        with torch.no_grad():
            if self.device == 'cuda':
                with torch.cuda.amp.autocast(enabled=True):
                    preds = self.model(prep_data['pillars'], prep_data['coords'], prep_data['num_points'], batch_size=1)
            else:
                preds = self.model(prep_data['pillars'], prep_data['coords'], prep_data['num_points'], batch_size=1)

        detections = decode_predictions(preds, self.cfg, score_threshold=0.4)

        if len(detections) > 0:
            active_tracks = self.tracker.update(detections)
        else:
            active_tracks = self.tracker.update(np.empty((0, 9)))

        self.publish_markers(active_tracks, msg.header)

        latency = (time.perf_counter() - start_time) * 1000
        self.get_logger().info(
            f"Spracované: {latency:.1f} ms | Detekcie: {len(detections)} | Aktívne stopy: {len(active_tracks)}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LiDARRealtimeNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Prijatý signál na ukončenie (Ctrl+C). Vypínam Edge Uzol...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()