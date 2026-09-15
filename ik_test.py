import time
import numpy as np
import mujoco
import mujoco.viewer

# Load model and allocate data memory
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

# Set a safe initial elbow-bent posture to prevent starting in a singularity
data.qpos[:7] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
data.ctrl[:7] = data.qpos[:7]
mujoco.mj_forward(model, data)

# Target 3D coordinate (centered in workspace)
target_pos = np.array([0.4, 0.0, 0.25])

# Target orientation matrix (fingers pointing straight down)
R_target = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

# IK Hyperparameters
damping = 1e-2      # Lambda factor for DLS
step_size = 0.05    # Smooth gain step size
tol_pos = 1e-3      # 1 mm tolerance

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.2, 0.0, 0.4]
    viewer.cam.distance = 1.8

    while viewer.is_running():
        # 1. Position Error (3D)
        current_pos = data.xpos[hand_id]
        pos_error = target_pos - current_pos
        
        # 2. Orientation Error (3D)
        R_curr = data.xmat[hand_id].reshape(3, 3)
        rot_error = 0.5 * (
            np.cross(R_curr[:, 0], R_target[:, 0]) +
            np.cross(R_curr[:, 1], R_target[:, 1]) +
            np.cross(R_curr[:, 2], R_target[:, 2])
        )
        
        # 6D Spatial error
        error_6d = np.hstack([pos_error, rot_error])
        
        # Iterate if either position or rotation is outside tolerance
        if np.linalg.norm(pos_error) > tol_pos or np.linalg.norm(rot_error) > 0.05:
            # 3. Compute spatial Jacobians
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr, current_pos, hand_id)
            
            J_full = np.vstack([jacp[:, :7], jacr[:, :7]])
            
            # 4. Damped Least Squares Pseudo-Inverse
            n_rows = J_full.shape[0]
            dls_inv = J_full.T @ np.linalg.inv(J_full @ J_full.T + (damping**2) * np.identity(n_rows))
            
            dq = dls_inv @ error_6d
            
            # CRITICAL FIX: Clamp max joint movement per step to prevent physical explosions
            dq = np.clip(dq, -0.05, 0.05)
            
            # Update position controllers smoothly
            data.ctrl[:7] += step_size * dq
            
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)