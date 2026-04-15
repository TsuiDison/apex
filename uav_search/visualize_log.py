import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def visualize_trajectory(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    steps = data.get('step_data', [])
    start_pos = data['setup']['start_position']
    target_pos_real = data['setup']['target_position']
    
    # AirSim uses (x, y, z) in meters. 
    # But the step_data shows 'position' like [20, 20, 5], which are grid indices.
    # We need to map the target_pos_real (meters) back to grid coordinates for comparison.
    # From multiprocess_s.py: 
    # GRID_SCALE = 5.0
    # UAV_START_GRID_POS = [20, 20, 5]
    
    GRID_SCALE = 5.0
    UAV_START_GRID_POS = np.array([20, 20, 5])
    start_pos_real = np.array(start_pos)
    
    def real_to_grid(real_pos):
        # relative_pos = (real_pos - start_pos_real) / GRID_SCALE
        # grid_pos = relative_pos + UAV_START_GRID_POS
        return (np.array(real_pos) - start_pos_real) / GRID_SCALE + UAV_START_GRID_POS

    target_grid = real_to_grid(target_pos_real)
    
    path = []
    for s in steps:
        path.append(s['uav_pose_after_action']['position'])
    
    path = np.array(path)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2D XY Plot
    ax[0].plot(path[:, 0], path[:, 1], 'b-', label='UAV Path', marker='o', markersize=4)
    ax[0].plot(path[0, 0], path[0, 1], 'go', label='Start (20,20)')
    ax[0].plot(target_grid[0], target_grid[1], 'r*', markersize=12, label='Target (Forklift)')
    ax[0].plot(path[-1, 0], path[-1, 1], 'ro', label='End (Collision)')
    ax[0].set_title('2D Trajectory (XY Plane)')
    ax[0].set_xlabel('Grid X')
    ax[0].set_ylabel('Grid Y')
    ax[0].grid(True)
    ax[0].legend()
    
    # Altitude Plot
    ax[1].plot(range(len(path)), path[:, 2], 'g-', label='Altitude', marker='x')
    ax[1].axhline(y=target_grid[2], color='r', linestyle='--', label='Target Altitude')
    ax[1].set_title('Altitude Change')
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('Grid Z (Altitude)')
    ax[1].grid(True)
    ax[1].legend()
    
    plt.suptitle(f"Trajectory Visualization: {os.path.basename(json_path)}\nFinal Status: {data['episode_summary']['termination_reason']}")
    
    output_path = json_path.replace('.json', '_vis.png')
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_trajectory(sys.argv[1])
    else:
        print("Please provide the path to the experiment JSON file.")
