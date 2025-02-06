import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import cv2
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

@dataclass
class VisitorMetrics:
    visitor_id: str
    first_seen: datetime
    last_seen: datetime
    total_dwell_time: timedelta
    zones_visited: List[str]
    entry_point: str
    exit_point: Optional[str] = None

class TrafficAnalytics:
    def __init__(self, store_dimensions: Tuple[int, int], zone_config: str = "zone_config.json"):
        self.width, self.height = store_dimensions
        self.heat_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.zone_config = zone_config
        self.zones = self.load_zone_config()
        
        # Tracking data structures
        self.visitor_metrics: Dict[str, VisitorMetrics] = {}
        self.zone_counters = defaultdict(int)
        self.hourly_traffic = defaultdict(int)
        self.current_positions = {}  # visitor_id -> (x, y, zone)
        
        # Analytics settings
        self.heat_map_decay = 0.99  # Decay factor for heat map
        self.min_dwell_time = timedelta(seconds=30)  # Minimum time to count as a visit
        
        # Create analytics directory if it doesn't exist
        os.makedirs("analytics", exist_ok=True)

    def load_zone_config(self) -> Dict[str, List[Tuple[int, int]]]:
        """Load store zone configuration."""
        try:
            with open(self.zone_config, 'r') as f:
                config = json.load(f)
                return {zone['name']: [tuple(point) for point in zone['points']] 
                       for zone in config['zones']}
        except FileNotFoundError:
            print(f"Zone config file {self.zone_config} not found. Creating default.")
            self.create_default_zone_config()
            return self.load_zone_config()

    def create_default_zone_config(self):
        """Create a default zone configuration file."""
        default_config = {
            "zones": [
                {
                    "name": "entrance",
                    "points": [[0, 0], [100, 0], [100, 100], [0, 100]]
                }
            ]
        }
        with open(self.zone_config, 'w') as f:
            json.dump(default_config, f, indent=4)

    def update_visitor_position(self, visitor_id: str, position: Tuple[int, int], camera_id: str, timestamp: datetime):
        """Update visitor position and related metrics."""
        x, y = position
        current_zone = self.get_zone_for_position(position)
        
        # Update heat map
        if 0 <= y < self.height and 0 <= x < self.width:
            self.heat_map[y, x] += 1.0
        
        # Apply decay to heat map
        self.heat_map *= self.heat_map_decay
        
        # Update visitor metrics
        if visitor_id not in self.visitor_metrics:
            # New visitor
            self.visitor_metrics[visitor_id] = VisitorMetrics(
                visitor_id=visitor_id,
                first_seen=timestamp,
                last_seen=timestamp,
                total_dwell_time=timedelta(0),
                zones_visited=[current_zone] if current_zone else [],
                entry_point=camera_id
            )
        else:
            metrics = self.visitor_metrics[visitor_id]
            metrics.last_seen = timestamp
            metrics.total_dwell_time = metrics.last_seen - metrics.first_seen
            
            if current_zone and current_zone not in metrics.zones_visited:
                metrics.zones_visited.append(current_zone)
        
        # Update current position
        self.current_positions[visitor_id] = (x, y, current_zone)
        
        # Update hourly traffic
        hour_key = timestamp.strftime("%Y-%m-%d %H:00")
        self.hourly_traffic[hour_key] += 1

    def get_zone_for_position(self, position: Tuple[int, int]) -> Optional[str]:
        """Determine which zone a position falls into."""
        x, y = position
        for zone_name, points in self.zones.items():
            if cv2.pointPolygonTest(np.array(points), (x, y), False) >= 0:
                return zone_name
        return None

    def generate_heat_map_visualization(self) -> np.ndarray:
        """Generate a visualization of the current heat map."""
        normalized = cv2.normalize(self.heat_map, None, 0, 255, cv2.NORM_MINMAX)
        heat_map_color = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Draw zone boundaries
        for zone_name, points in self.zones.items():
            points_array = np.array(points, dtype=np.int32)
            cv2.polylines(heat_map_color, [points_array], True, (255, 255, 255), 2)
            
            # Add zone label
            centroid = np.mean(points_array, axis=0, dtype=np.int32)
            cv2.putText(heat_map_color, zone_name, tuple(centroid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return heat_map_color

    def save_analytics_report(self):
        """Generate and save analytics report."""
        current_time = datetime.now()
        report = {
            "timestamp": current_time.isoformat(),
            "total_visitors": len(self.visitor_metrics),
            "active_visitors": len(self.current_positions),
            "zone_statistics": dict(self.zone_counters),
            "average_dwell_time": str(self.calculate_average_dwell_time()),
            "peak_hours": self.get_peak_hours()
        }
        
        # Save report
        report_file = f"analytics/traffic_report_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Save heat map visualization
        heat_map_file = f"analytics/heat_map_{current_time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(heat_map_file, self.generate_heat_map_visualization())
        
        # Generate and save traffic graph
        self.generate_traffic_graph()

    def calculate_average_dwell_time(self) -> timedelta:
        """Calculate average dwell time of visitors."""
        if not self.visitor_metrics:
            return timedelta(0)
        
        total_time = sum((m.total_dwell_time for m in self.visitor_metrics.values()),
                        start=timedelta(0))
        return total_time / len(self.visitor_metrics)

    def get_peak_hours(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get the top N peak hours by traffic."""
        sorted_hours = sorted(self.hourly_traffic.items(),
                            key=lambda x: x[1],
                            reverse=True)
        return sorted_hours[:top_n]

    def generate_traffic_graph(self):
        """Generate and save traffic over time graph."""
        if not self.hourly_traffic:
            return
        
        # Convert to pandas for easier plotting
        df = pd.DataFrame.from_dict(self.hourly_traffic, orient='index',
                                  columns=['count']).sort_index()
        
        plt.figure(figsize=(12, 6))
        df.plot(kind='line', marker='o')
        plt.title('Store Traffic Over Time')
        plt.xlabel('Time')
        plt.ylabel('Visitor Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        current_time = datetime.now()
        plt.savefig(f"analytics/traffic_graph_{current_time.strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

    def clear_old_data(self, max_age: timedelta = timedelta(days=7)):
        """Clear visitor data older than specified age."""
        current_time = datetime.now()
        to_remove = []
        
        for visitor_id, metrics in self.visitor_metrics.items():
            if current_time - metrics.last_seen > max_age:
                to_remove.append(visitor_id)
        
        for visitor_id in to_remove:
            del self.visitor_metrics[visitor_id]
            if visitor_id in self.current_positions:
                del self.current_positions[visitor_id]

if __name__ == "__main__":
    # Example usage
    analytics = TrafficAnalytics((800, 600))
    
    # Simulate some visitor movements
    current_time = datetime.now()
    analytics.update_visitor_position("visitor1", (100, 100), "cam1", current_time)
    analytics.update_visitor_position("visitor2", (200, 300), "cam2", current_time)
    
    # Generate report
    analytics.save_analytics_report()
