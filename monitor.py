import os
import time
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import inference  # import our inference functions

# Folder to watch
WATCH_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Mahasat_therm-2')

# Where we'll store the latest prediction info
LATEST_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latest_prediction.json')

class ThermImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
            print(f"[monitor.py] New image detected: {event.src_path}")
            # Run inference to get temperature map
            temp_map = inference.predict_temperature_map(event.src_path)
            if temp_map is None:
                return
            
            # For demonstration, let's store the average temperature as the "prediction"
            avg_temp = float(np.mean(temp_map))

            # You could store the entire temp_map if you want, but it might be large.
            # We'll store a single statistic + the path + a timestamp
            data = {
                "image_path": event.src_path,
                "avg_temperature": avg_temp,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }

            # Save to JSON
            with open(LATEST_JSON, 'w') as f:
                json.dump(data, f)
            print("[monitor.py] Updated latest_prediction.json:", data)

if __name__ == "__main__":
    from watchdog.observers import Observer

    if not os.path.exists(WATCH_FOLDER):
        raise ValueError(f"Watch folder does not exist: {WATCH_FOLDER}")

    event_handler = ThermImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_FOLDER, recursive=True)
    observer.start()

    print(f"[monitor.py] Monitoring {WATCH_FOLDER} for new thermal images...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()