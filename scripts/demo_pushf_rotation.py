import os
import sys
import imageio
import numpy as np

# Ensure repo root on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_policy.env.pushf.pushf_image_env import PushFImageEnv


def main(out_path: str = "data/pushf/pushf_rotation_demo.mp4", steps: int = 120):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    env = PushFImageEnv(render_size=256)
    obs = env.reset()

    # Simple policy: move agent target around the block to induce rotation toward goal
    def wrap_pi(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    writer = imageio.get_writer(out_path, fps=10)
    try:
        for t in range(steps):
            # Extract block and goal pose
            # Construct info-like fields using underlying PushFEnv attributes
            info_like = {
                "pos_agent": np.array(env.agent.position),
                "block_pose": np.array(list(env.block.position) + [env.block.angle]),
                "goal_pose": env.goal_pose,
            }
            block_pos = info_like["block_pose"][:2]
            block_theta = info_like["block_pose"][2]
            goal_pos = info_like["goal_pose"][:2]
            goal_theta = info_like["goal_pose"][2]

            # Direction to goal and torque side based on angle error
            dir_vec = goal_pos - block_pos
            dist = np.linalg.norm(dir_vec) + 1e-6
            dir_vec = dir_vec / dist
            ang_err = wrap_pi(goal_theta - block_theta)
            perp = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float32)

            # Target point: ahead plus lateral offset to rotate
            tgt = block_pos + dir_vec * 45.0 + perp * (35.0 * np.sign(ang_err))
            tgt = np.clip(tgt, 10, 502)

            obs, r, done, info = env.step(tgt)
            frame = env.render(mode="rgb_array")
            writer.append_data(frame)
            if done:
                break
    finally:
        writer.close()
    print(f"Saved demo to {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/pushf/pushf_rotation_demo.mp4"
    main(out)
