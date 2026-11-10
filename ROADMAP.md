1. Custom Architecture, with CNN and CNN-outputs + scalars plus history (RNN history or frame-stacking)
2. Make CNN pre-loadable in the custom architecture, trainable and verifiable offline with beams
3. Make waypoint planning possible with code-based waypoint generation and observation space additions, rewarding when clearing waypoints
Observation space new elements planned:
- cumulative angle change over the upcoming 2 waypoints of the route
- cumulative angle change over the waypoints 3-5 of the route
- cumulative angle change over the waypoints 5-10 of the route
- attack angle on centerline (difference with the -y car axis representing frontal direction)
- world position
- distance to first waypoint
4. (Maybe) Invalid action masking
