"""Wildlife Detection System.

A fully-local, motion-triggered wildlife detection package. It subscribes to
PoE camera detection events, grabs a short RTSP frame burst on each event, runs
YOLO object detection on Apple Silicon (MPS), and persists only frames that
contain an animal above a configurable confidence threshold. A small read-only
Flask gallery serves the captures on the LAN.

No cloud, no subscriptions -- everything runs on the host Mac.
"""

__version__ = "0.1.0"
