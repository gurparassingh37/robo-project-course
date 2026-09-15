import time
import numpy as np
import mujoco
import mujoco.viewer

# 1. Load Simulation Model and Data Memory
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

# 2. Initialize Joint Positions to an Elbow-Up Posture
q_nominal = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
data.qpos[:7] = q_nominal
data.ctrl[:7] = q_nominal
mujoco.mj_forward(model, data)

# 3. Extract Valid Top-Down Rotation Matrix from Initial Pose
R_target = data.xmat[hand_id].reshape(3, 3).copy()

# 4. Define Key 3D Workspace Waypoints [X, Y, Z]
key_waypoints = [
    np.array([0.45,  0.15, 0.30]),  # Waypoint 0: Hover above Pick
    np.array([0.45,  0.15, 0.14]),  # Waypoint 1: Lower to Pick
    np.array([0.45,  0.15, 0.30]),  # Waypoint 2: Lift straight up
    np.array([0.45, -0.15, 0.30]),  # Waypoint 3: Transport across Workspace
    np.array([0.45, -0.15, 0.14]),  # Waypoint 4: Lower to Place
    np.array([0.45, -0.15, 0.30]),  # Waypoint 5: Retract straight up
]

phase_names = [
    "1/6: Hovering above Pick Position",
    "2/6: Lowering to Pick Object",
    "3/6: Lifting Object Up",
    "4/6: Transporting across Workspace",
    "5/6: Lowering to Place Position",
    "6/6: Retracting Up (Execution Complete)"
]

# 5. Build Interpolated Dense 3D Straight Path (120 micro-steps per segment)
steps_per_segment = 120
dense_path = []
segment_ends = []

for i in range(len(key_waypoints) - 1):
    segment = np.linspace(key_waypoints[i], key_waypoints[i+1], steps_per_segment)
    dense_path.extend(segment)
    segment_ends.append(len(dense_path) - 1)

dense_path = np.array(dense_path)
total_frames = len(dense_path)

# 6. Controller Constants
damping = 1e-2
k_null = 0.8  # Nullspace posture restoration gain
frame_idx = 0
current_phase = 0

print("\n--- Starting High-Precision Pick-and-Place Trajectory ---")
print(f"Active Phase -> {phase_names[0]}")

# 7. Main Control Loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.45, 0.0, 0.25]
    viewer.cam.distance = 1.5

    while viewer.is_running():
        # Step A: Get active micro-target position
        target_pos = dense_path[frame_idx]

        # Step B: Calculate 3D position error
        current_pos = data.xpos[hand_id]
        pos_error = target_pos - current_pos

        # Step C: Calculate orientation error matrix
        R_curr = data.xmat[hand_id].reshape(3, 3)
        rot_error = 0.5 * (
            np.cross(R_curr[:, 0], R_target[:, 0]) +
            np.cross(R_curr[:, 1], R_target[:, 1]) +
            np.cross(R_curr[:, 2], R_target[:, 2])
        )
        
        # Combine into 6D Cartesian Error Vector
        error_6d = np.hstack([pos_error, rot_error * 0.5])

        # Step D: Compute 6x7 Spatial Jacobian Matrix
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, current_pos, hand_id)
        J_full = np.vstack([jacp[:, :7], jacr[:, :7]])

        # Step E: Inverse Kinematics via Damped Least Squares (DLS)
        n_rows = J_full.shape[0]
        dls_inv = J_full.T @ np.linalg.inv(J_full @ J_full.T + (damping**2) * np.identity(n_rows))

        # Step F: Joint velocities for main task + Nullspace posture lock
        dq_task = dls_inv @ error_6d
        I_7 = np.identity(7)
        P_null = I_7 - (dls_inv @ J_full)
        dq_null = k_null * (q_nominal - data.qpos[:7])

        dq = dq_task + (P_null @ dq_null)
        dq = np.clip(dq, -0.05, 0.05)  # Enforce realistic speed boundary

        # Step G: Direct State Update (Eliminates controller windup)
        q_next = data.qpos[:7] + dq
        data.qpos[:7] = q_next
        data.ctrl[:7] = q_next

        # Step H: Physics Step and Render
        mujoco.mj_step(model, data)
        viewer.sync()

        # Step I: Bounds-Checked Terminal Tracking Logic
        if current_phase < len(segment_ends) and frame_idx == segment_ends[current_phase]:
            print(f"Phase Complete -> {phase_names[current_phase]}")
            time.sleep(0.3)  # Visual pause at key locations
            current_phase += 1
            if current_phase < len(phase_names):
                print(f"Advancing Phase -> {phase_names[current_phase]}")

        # Step J: Increment frame index along dense path
        if frame_idx < total_frames - 1:
            frame_idx += 1

        time.sleep(model.opt.timestep)