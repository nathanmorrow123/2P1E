import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

# Parameters
num_pursuers = 2
dt = 0.1  # Time step
noise_level = 0.0001  # Noise level
capture_radius = 1  # Radius within which the evader is considered captured
trail_length = 10  # Length of the ship trails (number of previous positions to show)
v_E = np.sqrt(2)
v_P = 1

# Initial positions
pursuer_positions = np.array([[2.0, 10.0], [-2.0, 10.0]])
evader_position = np.array([0.0, 0.0])

# Store positions over time for animation
frames = 250  # Number of animation frames
pursuer_positions_over_time = np.zeros((frames, num_pursuers, 2))
evader_positions_over_time = np.zeros((frames, 2))

# Capture status
captured_by_pursuer = [False] * num_pursuers
capture_buffer_counter = 0

# Populate positions over time
for frame in range(frames):
    if any(captured_by_pursuer):
        if capture_buffer_counter < 10:
            capture_buffer_counter += 1
        else:
            break
    else:
        for i in range(num_pursuers):
            if not captured_by_pursuer[i]:
                if np.linalg.norm(pursuer_positions[i] - evader_position) <= capture_radius:
                    captured_by_pursuer[i] = True
        
        if not all(captured_by_pursuer):
            midpoint = np.mean(pursuer_positions, axis=0)
            pursuer_vector = np.diff(pursuer_positions, axis=0)[0]
            perpendicular_direction = np.array([-pursuer_vector[1], -pursuer_vector[0]])
            evader_direction = perpendicular_direction / np.linalg.norm(perpendicular_direction)
            noise = noise_level * np.random.randn(2)
            evader_position += v_E * evader_direction * dt + noise
        evader_positions_over_time[frame] = evader_position.copy()
        
        for i in range(num_pursuers):
            direction = evader_position - pursuer_positions[i]
            noise = noise_level * np.random.randn(2)
            pursuer_positions[i] += v_P * direction / np.linalg.norm(direction) * dt + noise
            pursuer_positions_over_time[frame, i] = pursuer_positions[i].copy()

# Create capture circles
capture_circles = []
theta = np.linspace(0, 2 * np.pi, 100)
for pos in pursuer_positions:
    x_circle = capture_radius * np.cos(theta) + pos[0]
    y_circle = capture_radius * np.sin(theta) + pos[1]
    capture_circles.append((x_circle, y_circle))

# Create animation frames
frames_data = []
for frame in range(frames):
    frame_data = []
    
    # Add evader position
    frame_data.append(
        go.Scatter(
            x=[evader_positions_over_time[frame, 0]],
            y=[evader_positions_over_time[frame, 1]],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Evader'
        )
    )
    
    # Add pursuer positions and trails
    for i in range(num_pursuers):
        frame_data.append(
            go.Scatter(
                x=[pursuer_positions_over_time[frame, i, 0]],
                y=[pursuer_positions_over_time[frame, i, 1]],
                mode='markers',
                marker=dict(size=10, color='blue'),
                name=f'Pursuer {i+1}'
            )
        )
        # Add pursuer trail
        if frame >= trail_length:
            frame_data.append(
                go.Scatter(
                    x=pursuer_positions_over_time[frame-trail_length:frame, i, 0],
                    y=pursuer_positions_over_time[frame-trail_length:frame, i, 1],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    opacity=0.2,
                    showlegend=False
                )
            )
    
    # Add evader trail
    if frame >= trail_length:
        frame_data.append(
            go.Scatter(
                x=evader_positions_over_time[frame-trail_length:frame, 0],
                y=evader_positions_over_time[frame-trail_length:frame, 1],
                mode='lines',
                line=dict(color='red', width=2),
                opacity=0.2,
                showlegend=False
            )
        )
    
    # Add capture circles
    for x_circle, y_circle in capture_circles:
        frame_data.append(
            go.Scatter(
                x=x_circle,
                y=y_circle,
                mode='lines',
                line=dict(color='darkblue', width=1, dash='dash'),
                showlegend=False
            )
        )
    
    frames_data.append(go.Frame(data=frame_data))

# Layout
fig = go.Figure(
    data=frames_data[0].data,
    frames=frames_data,
    layout=go.Layout(
        title='Pursuers and Fugitive Problem',
        xaxis=dict(range=[-5, 5], title='X'),
        yaxis=dict(range=[0, 20], title='Y'),
        updatemenus=[dict(
            type='buttons',
            y=0.2,
            x=1.05,
            active=0,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(
                    frame=dict(duration=50, redraw=True),
                    transition=dict(duration=0),
                    fromcurrent=True,
                    mode='immediate'
                )]
            )]
        )]
    )
)

# Save and display the animation
pio.write_html(fig, file='pursuers_and_fugitive_2d.html', auto_open=True)
