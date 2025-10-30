import os
import argparse
import numpy as np
import zarr
from tqdm import trange

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pushf.pushf_image_env import PushFImageEnv


def collect(output_path: str, n_episodes: int = 20, max_steps: int = 300, render=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    store = zarr.DirectoryStore(output_path)
    rb = ReplayBuffer.create_empty_zarr(storage=store)

    env = PushFImageEnv(render_size=96)
    # By default use a scripted (non-interactive) agent so remote collection can run
    # without human input. The caller may override to use teleop if desired.
    teleop = None

    for ep in range(n_episodes):
        obs = env.reset()
        imgs = []
        states = []
        actions = []
        done = False
        for t in trange(max_steps, desc=f"Episode {ep+1}/{n_episodes}"):
            # render human window if requested
            if render:
                env.render(mode="rgb_array")

            act = teleop.act(obs)
            if act is None:
                act = env.agent.position  # hold position if not grabbing
            obs, reward, done, info = env.step(act)

            # record
            img = env.render(mode="rgb_array")  # H,W,C uint8
            state = np.array(list(info["pos_agent"]) + list(info["block_pose"]))
            imgs.append(img)
            states.append(state)
            actions.append(np.array(act))

            if done:
                break

        # to arrays
        imgs = np.asarray(imgs, dtype=np.uint8)
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)

        # add episode to replay buffer
        rb.add_episode({
            'img': imgs,        # T, H, W, C
            'state': states,    # T, 5 (agent_x, agent_y, block_x, block_y, theta)
            'action': actions   # T, 2
        })

    # finalize
    rb.save_to_path(output_path)
    print(f"Saved {n_episodes} episodes to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/pushf/pushf_replay.zarr")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--agent", type=str, choices=["teleop", "scripted"], default="scripted",
                        help="Agent type to use for collection: 'teleop' for interactive control, 'scripted' for non-interactive scripted policy")
    args = parser.parse_args()

    # Select agent type
    use_scripted = args.agent == "scripted"
    collect(output_path=args.out, n_episodes=args.episodes, max_steps=args.steps, render=not args.no_render)
