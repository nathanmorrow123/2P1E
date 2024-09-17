import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

# Parameters
num_cutters = 2
dt = 0.1  # Time step in seconds
noise_level = 0.001  # Noise level
frames = 200  # Number of animation frames

# Initial positions
cutter_positions = np.random.uniform(low=-50, high=-30, size=(num_cutters, 3))  # Pursuers start much further south
cutter_positions[:, 1] = np.random.uniform(low=-5, high=5, size=(num_cutters,))  # Randomize the east-west positions
evader_position = np.array([0, 0, 10])  # Evader starts at a higher altitude, directly north

# Store positions over time for animation
cutter_positions_over_time = np.zeros((frames, num_cutters, 3))
evader_positions_over_time = np.zeros((frames, 3))

# Populate positions over time
for frame in range(frames):
    # Update evader position: parabolic trajectory southward
    evader_position[0] += -0.2  # Southward movement (negative X direction)
    evader_position[2] = 0.1 * (evader_position[0] ** 2) + 10  # Parabolic descent
    evader_positions_over_time[frame] = evader_position.copy()
    
    for i in range(num_cutters):
        direction = evader_position - cutter_positions[i]
        noise = noise_level * np.random.randn(3)
        cutter_positions[i] += 0.6 * direction / np.linalg.norm(direction) * dt + noise
        cutter_positions_over_time[frame, i] = cutter_positions[i].copy()

# Create animation frames with elapsed time display
animation_frames = []
for frame in range(frames):
    # Calculate time elapsed
    time_elapsed = frame * dt
    
    frame_data = [
        go.Scatter3d(
            x=evader_positions_over_time[:frame+1, 0],
            y=evader_positions_over_time[:frame+1, 1],
            z=evader_positions_over_time[:frame+1, 2],
            mode='lines+markers',
            marker=dict(size=5, color='red'),
            line=dict(color='red', width=2),
        )
    ]
    for i in range(num_cutters):
        frame_data.append(
            go.Scatter3d(
                x=cutter_positions_over_time[:frame+1, i, 0],
                y=cutter_positions_over_time[:frame+1, i, 1],
                z=cutter_positions_over_time[:frame+1, i, 2],
                mode='lines+markers',
                marker=dict(size=5, color='blue'),
                line=dict(color='blue', width=2),
            )
        )
    
    # Update layout with elapsed time
    layout_update = dict(
        annotations=[
            dict(
                text=f'Time elapsed: {time_elapsed:.1f} seconds',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0,
                y=1.05,
                font=dict(size=16, color="white")
            )
        ]
    )
    
    animation_frames.append(go.Frame(data=frame_data, layout=layout_update))

# Layout
layout = go.Layout(
    title='Two Cutters and Fugitive Ship Problem',
    scene=dict(
        xaxis=dict(range=[-55, 5]),
        yaxis=dict(range=[-10, 10]),
        zaxis=dict(range=[0, 40]),
        aspectmode='cube',
    ),
    updatemenus=[dict(type='buttons', showactive=False,
                      buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])])]
)

# Create figure
fig = go.Figure(data=animation_frames[0].data, layout=layout, frames=animation_frames)

# Render and save
pio.write_html(fig, file='cutters_and_fugitive_3d.html', auto_open=True)
