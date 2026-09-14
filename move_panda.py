import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 1. Visualize joint axis arrows (Blue arrows)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # 2. Visualize local body coordinate frames (RGB tripods)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

    # Camera positioning
    viewer.cam.lookat[:] = [0.0, 0.0, 0.3]
    viewer.cam.distance = 1.3
    viewer.cam.elevation = -15.0
    viewer.cam.azimuth = 90.0

    start_time = time.time()

    while viewer.is_running():
        t = time.time() - start_time

        # Drive Joint 1
        data.ctrl[0] = np.sin(t * 2.0) * 1.5

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)