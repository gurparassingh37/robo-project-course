import time
import numpy as np
import mujoco
import mujoco.viewer

# Load model and allocate data memory
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Lookup body ID for the end-effector (hand)
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Render body RGB frames to visually track hand transformation
    

    # Set camera placement
    viewer.cam.lookat[:] = [0.0, 0.0, 0.4]
    viewer.cam.distance = 1.5
    viewer.cam.elevation = -20.0
    viewer.cam.azimuth = 90.0

    start_time = time.time()
    last_print = 0

    while viewer.is_running():
        t = time.time() - start_time

        # Drive Joint 1 (base turn) and Joint 4 (elbow pitch) dynamically
        data.ctrl[0] = np.sin(t * 1.5) * 0.8
        data.ctrl[3] = -2.0 + np.sin(t * 2.0) * 0.5

        # Advance physics (evaluates FK transformations automatically)
        mujoco.mj_step(model, data)
        viewer.sync()

        # Print updated FK coordinates 5 times per second
        if t - last_print > 0.2:
            gripper_pos = data.xpos[hand_id]
            print(f"FK Hand Position -> X: {gripper_pos[0]:.3f}m | Y: {gripper_pos[1]:.3f}m | Z: {gripper_pos[2]:.3f}m")
            last_print = t

        time.sleep(model.opt.timestep)