import json
import numpy as np

# Load the segments data
with open('pipeline_workspace/data/segmentation/0_segments_stage2.json', 'r') as f:
    data = json.load(f)

# Get segments 119-123
target_segments = []
for i in range(119, 124):
    if i < len(data):
        target_segments.append(data[i])

print('Checking oscillation pattern for segments 119-123:')

# Show actual segment details
for i, seg in enumerate(target_segments, 119):
    start_freq = seg['start_pitch']['frequency']
    end_freq = seg['end_pitch']['frequency'] 
    print(f'Segment {i}: {start_freq:.1f} Hz -> {end_freq:.1f} Hz')

print()

# Check oscillation using actual endpoints, not averages
endpoint_frequencies = []
for seg in target_segments:
    endpoint_frequencies.append(seg['start_pitch']['frequency'])
# Add the final endpoint
endpoint_frequencies.append(target_segments[-1]['end_pitch']['frequency'])

print('Endpoint frequencies:', [round(f, 1) for f in endpoint_frequencies])

# Calculate center frequency using endpoints
center_freq = (max(endpoint_frequencies) + min(endpoint_frequencies)) / 2.0
print(f'Center frequency: {center_freq:.1f} Hz')

# Classify each segment as above or below center
positions = ['high' if freq > center_freq else 'low' for freq in endpoint_frequencies]
print('Positions:', positions)

# Count transitions
transitions = 0
for i in range(1, len(positions)):
    if positions[i] != positions[i-1]:
        transitions += 1

print('Transitions:', transitions)
print('Needs >= 4 transitions for vibrato:', transitions >= 4)