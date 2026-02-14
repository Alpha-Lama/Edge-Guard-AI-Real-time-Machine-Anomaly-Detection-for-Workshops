Why AI & Computer Vision

EdgeGuard leverages AI-powered computer vision to automate continuous safety monitoring—a task impossible for humans to perform 24/7 across multiple machines. Traditional manual inspections are periodic, subjective, and miss real-time hazards. Our system uses OpenCV's edge detection algorithms and gradient analysis to identify anomalous patterns that indicate safety risks, providing objective, quantitative metrics updated every second.

Workflow

Video feeds from workshop cameras flow through a processing pipeline: frames are captured and preprocessed, then analyzed using multi-scale Canny edge detection at three granularities. Sobel gradient analysis extracts edge strength and orientation. Spatial clustering identifies localized anomaly patterns. Quality metrics (density, sharpness, orientation balance) are calculated and compared against camera-specific baselines. When deviations exceed thresholds, alerts are generated with severity scoring and dispatched to operators via a real-time dashboard.


Built on Node.js for async I/O efficiency, using OpenCV.js for computer vision processing. Redis Streams handles frame distribution across processing pods. InfluxDB stores time-series metrics. AWS S3 archives frame snapshots. The system deploys on AWS EKS with Kubernetes HorizontalPodAutoscaler for elastic scaling based on camera load. React dashboard with WebSocket provides sub-2-second alert delivery. Prometheus and Grafana enable observability.


Reactive vs. Proactive Safety: Eliminates reliance on post-incident analysis by detecting anomalies in real-time before accidents occur.

Scalability: Manual monitoring doesn't scale beyond a few machines. EdgeGuard handles 20+ cameras per pod with autoscaling to 10 pods (200+ cameras).

Consistency: Human attention degrades over time. The system maintains consistent vigilance with sub-500ms latency.

Data-Driven Insights: Provides quantitative metrics and historical trends for root cause analysis and continuous improvement, replacing subjective assessments.

Cost Efficiency: Reduces need for dedicated safety monitors while improving coverage and response times.
