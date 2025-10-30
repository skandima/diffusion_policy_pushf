import gym
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import shapely.ops as so
from shapely.errors import TopologicalError
import cv2
import skimage.transform as st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions


def pymunk_to_shapely(body, shapes):
    """
    Convert pymunk Poly shapes attached to a body into a single valid Shapely geometry.
    Repairs invalid polygons via buffer(0) and unions overlapping parts.
    """
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            if len(verts) >= 3:
                poly = sg.Polygon(verts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                geoms.append(poly)
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    if len(geoms) == 0:
        return sg.GeometryCollection()
    geom = so.unary_union(geoms)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


class PushFEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_action=True,
        render_size=96,
        reset_to_state=None,
    ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20  # PD control
        self.control_hz = self.metadata["video.frames_per_second"]
        # legacy set_state for data compatibility
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float64),
            high=np.array([ws, ws, ws, ws, np.pi * 2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64,
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float64),
            high=np.array([ws, ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array(
                [
                    rs.randint(50, 450),
                    rs.randint(50, 450),
                    rs.randint(100, 400),
                    rs.randint(100, 400),
                    rs.randn() * 2 * np.pi - np.pi,
                ]
            )
        self._set_state(state)

        observation = self._get_obs()
        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for _ in range(n_steps):
                # PD control towards action target
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                    Vec2d(0, 0) - self.agent.velocity
                )
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward based on overlap with goal pose
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        # Robust intersection area computation
        try:
            intersection_area = goal_geom.intersection(block_geom).area
        except TopologicalError:
            # Attempt repair and retry
            try:
                intersection_area = goal_geom.buffer(0).intersection(block_geom.buffer(0)).area
            except Exception:
                intersection_area = 0.0
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area if goal_area > 0 else 0.0
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(
                Vec2d(*pygame.mouse.get_pos()), self.screen
            )
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) + tuple(self.block.position) + (self.block.angle % (2 * np.pi),)
        )
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        # approximate inertia using a box; exact value not critical
        inertia = pymunk.moment_for_box(mass, (60, 120))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [
                pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface)
                for v in shape.get_vertices()
            ]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action and (self.latest_action is not None):
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(
                img,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        if self.legacy:
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block
        self.space.step(1.0 / self.sim_hz)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(translation=self.goal_pose[:2], rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2],
        )
        tf_img_new = st.AffineTransform(matrix=tf_img_obj.params @ tf_obj_new.params)
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(list(agent_pos_new[0]) + list(tf_img_new.translation) + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_f((256, 300), 0)
        self.goal_color = pygame.Color("LightGreen")
        self.goal_pose = np.array([256, 256, np.pi / 6])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 60 * 120
        self.success_threshold = 0.95  # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def add_f(self, position, angle, scale=30, color="LightSlateGray", mask=pymunk.ShapeFilter.ALL_MASKS()):
        """
        Construct an 'F' shape using three rectangles:
        - Vertical stem (height=length*scale, width=scale)
        - Top bar (width=length*scale, height=scale)
        - Middle bar (width=length*scale*0.7, height=scale)
        """
        mass = 1
        length = 4
        stem_w = scale
        stem_h = length * scale
        top_w = length * scale
        top_h = scale
        mid_w = int(length * scale * 0.7)
        mid_h = scale

        # Define polygons centered around (0,0), then position body
        # Vertical stem centered at (-(top_w - stem_w)/2, stem_h/2)
        stem_vertices = [
            (-stem_w / 2, 0),
            (stem_w / 2, 0),
            (stem_w / 2, stem_h),
            (-stem_w / 2, stem_h),
        ]
        # Top bar at top of stem
        top_vertices = [
            (0, stem_h - top_h),
            (top_w, stem_h - top_h),
            (top_w, stem_h),
            (0, stem_h),
        ]
        # Middle bar at ~half height
        mid_y = stem_h * 0.5
        mid_vertices = [
            (0, mid_y - mid_h / 2),
            (mid_w, mid_y - mid_h / 2),
            (mid_w, mid_y + mid_h / 2),
            (0, mid_y + mid_h / 2),
        ]

        inertia1 = pymunk.moment_for_poly(mass, vertices=stem_vertices)
        inertia2 = pymunk.moment_for_poly(mass, vertices=top_vertices)
        inertia3 = pymunk.moment_for_poly(mass, vertices=mid_vertices)
        body = pymunk.Body(mass, inertia1 + inertia2 + inertia3)
        shape1 = pymunk.Poly(body, stem_vertices)
        shape2 = pymunk.Poly(body, top_vertices)
        shape3 = pymunk.Poly(body, mid_vertices)
        for s in (shape1, shape2, shape3):
            s.color = pygame.Color(color)
            s.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            (shape1.center_of_gravity + shape2.center_of_gravity + shape3.center_of_gravity) / 3
        )
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2, shape3)
        return body
