import json
import os
import time
from datetime import datetime
import numpy as np

class IncidentLogger:
    def __init__(self, log_dir="incidents", buffer_size=30):
        self.log_dir = log_dir
        self.buffer_size = buffer_size
        self.buffer = []  # List posledných N framov
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def add_frame(self, tracks):
        """Pridá aktuálny stav všetkých trackov do dočasného buffera."""
        frame_data = []
        for t in tracks:
            frame_data.append({
                "id": t.id,
                "class_id": t.class_id,
                "pos": t.kf.x[:3].tolist(),
                "vel": t.kf.x[7:9].tolist(),
                "risk": round(float(t.smoothed_risk), 3),
                "timestamp": time.time()
            })
        
        self.buffer.append(frame_data)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def save_incident(self, trigger_track_id, risk_value):
        """Uloží celý obsah buffera ako incident."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"incident_{timestamp_str}_ID{trigger_track_id}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        incident_report = {
            "incident_metadata": {
                "trigger_id": trigger_track_id,
                "max_risk": round(risk_value, 3),
                "recorded_at": timestamp_str,
                "buffer_length": len(self.buffer)
            },
            "data": self.buffer
        }
        
        with open(filepath, 'w') as f:
            json.dump(incident_report, f, indent=4)
            
        print(f"⚠️ KRITICKÁ UDALOSŤ ULOŽENÁ: {filepath}")
        # Po uložení vyčistíme buffer, aby sme neukladali ten istý incident duplicitne
        self.buffer = []
