import os
import argparse
import numpy as np
import zarr
from tqdm import trange

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pushf.pushf_image_env import PushFImageEnv


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def scripted_policy(info):
    # Simple heuristic: push towards goal position and add torque influence if angle error large
    agent_pos = info["pos_agent"]
    block_x, block_y, block_theta = info["block_pose"]
    goal_x, goal_y, goal_theta = info["goal_pose"]

    block_pos = np.array([block_x, block_y], dtype=np.float32)
    goal_pos = np.array([goal_x, goal_y], dtype=np.float32)

    pos_err = goal_pos - block_pos
    dist = np.linalg.norm(pos_err) + 1e-6
    dir_vec = pos_err / dist

    # base target: ahead of block towards goal
    tgt = block_pos + dir_vec * 35.0

    # angle correction: push on side to induce rotation
    ang_err = wrap_pi(goal_theta - block_theta)
    if abs(ang_err) > 0.15:
        # perpendicular to dir_vec to generate torque
        perp = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float32)
        tgt += perp * (30.0 * np.sign(ang_err))

    # clamp to arena bounds
    tgt = np.clip(tgt, 10, 502)
    return tgt


def collect(output_path: str, n_episodes: int = 20, max_steps: int = 300, render=True, mode="scripted"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    store = zarr.DirectoryStore(output_path)
    rb = ReplayBuffer.create_empty_zarr(storage=store)

    env = PushFImageEnv(render_size=96)
    teleop = env.teleop_agent()

    for ep in range(n_episodes):
        obs = env.reset()
        imgs = []
        states = []
        actions = []
        done = False
        for _ in trange(max_steps, desc=f"Episode {ep+1}/{n_episodes}"):
            if render:
                env.render(mode="rgb_array")

            # choose action
            if mode == "teleop":
                act = teleop.act(obs)
                if act is None:
                    act = env.agent.position
            else:
                # get last info by stepping zero movement for info at reset
                # Use last known info by querying internal state via a zero-time advance
                # Here we approximate by constructing info from env attributes
                info = {
                    "pos_agent": np.array(env.agent.position),
                    "block_pose": np.array(list(env.block.position) + [env.block.angle]),
                    "goal_pose": env.goal_pose,
                }
                act = scripted_policy(info)

            obs, reward, done, info = env.step(act)

            img = env.render(mode="rgb_array")
            state = np.array(list(info["pos_agent"]) + list(info["block_pose"]))
            imgs.append(img)
            states.append(state)
            actions.append(np.array(act))

            if done:
                break

        imgs = np.asarray(imgs, dtype=np.uint8)
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)

        rb.add_episode({
            "img": imgs,
            "state": states,
            "action": actions,
        })

    # If using a Zarr-backed buffer, data has already been written to disk via the DirectoryStore.
    # Saving to the same path would recreate the 'data' group and can wipe arrays when source==dest.
    if rb.backend != 'zarr':
        rb.save_to_path(output_path)
    print(f"Saved {n_episodes} episodes to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/pushf/pushf_replay.zarr")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--mode", type=str, choices=["scripted", "teleop"], default="scripted")
    args = parser.parse_args()

    collect(
        output_path=args.out,
        n_episodes=args.episodes,
        max_steps=args.steps,
        render=not args.no_render,
        mode=args.mode,
    )
