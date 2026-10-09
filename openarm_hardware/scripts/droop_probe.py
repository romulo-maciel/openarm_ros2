#!/usr/bin/env python3
# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Measure the drop of an arm between two trajectories.

Sends a list of joint goals to {side}_joint_trajectory_controller the same way
MoveIt does (first point = measured state at t=0), records /joint_states and
the controller reference, and reports for every move:

  sag_rest  reference - measured while holding before the move (PD error)
  drop      how far the joint goes against the motion direction right after
            the new trajectory starts, before it actually moves

Run with the stack up (demo.launch.py, joint_trajectory_controller):
  /usr/bin/python3 droop_probe.py --side left --csv /tmp/droop.csv
"""

import argparse
import csv
import math
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

# Elbow up (SRDF "hands_up"), shoulder alone with the elbow held, elbow down,
# elbow back up (the drop shows here), then back to zero.
DEFAULT_POSES = [
    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
    [0.3, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
    [0.3, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]


class Probe(Node):
    def __init__(self, side):
        super().__init__("droop_probe")
        self.joints = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        self.lock = threading.Lock()
        self.rows = []  # (t, q[7], effort[7], ref[7] or None)
        self.ref = None
        self.create_subscription(JointState, "/joint_states", self.on_js, 50)
        self.create_subscription(
            JointTrajectoryControllerState,
            f"/{side}_joint_trajectory_controller/controller_state",
            self.on_state, 50)
        self.client = ActionClient(
            self, FollowJointTrajectory,
            f"/{side}_joint_trajectory_controller/follow_joint_trajectory")

    def on_state(self, msg):
        idx = [msg.joint_names.index(j) for j in self.joints]
        with self.lock:
            self.ref = [msg.reference.positions[i] for i in idx]

    def on_js(self, msg):
        if not all(j in msg.name for j in self.joints):
            return
        idx = [msg.name.index(j) for j in self.joints]
        q = [msg.position[i] for i in idx]
        eff = [msg.effort[i] if msg.effort else math.nan for i in idx]
        with self.lock:
            self.rows.append((time.monotonic(), q, eff, self.ref))

    def latest(self):
        with self.lock:
            return self.rows[-1] if self.rows else None

    def move(self, goal_q, speed):
        now = self.latest()
        q0 = now[1]
        dist = max(abs(a - b) for a, b in zip(goal_q, q0))
        duration = max(1.0, dist / speed)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joints
        # MoveIt always starts the trajectory at the measured state.
        p0 = JointTrajectoryPoint(positions=list(q0), velocities=[0.0] * 7)
        p1 = JointTrajectoryPoint(positions=list(goal_q), velocities=[0.0] * 7)
        p1.time_from_start.sec = int(duration)
        p1.time_from_start.nanosec = int((duration % 1) * 1e9)
        goal.trajectory.points = [p0, p1]
        t_send = time.monotonic()
        fut = self.client.send_goal_async(goal)
        while not fut.done():
            time.sleep(0.005)
        res = fut.result().get_result_async()
        while not res.done():
            time.sleep(0.01)
        return t_send, q0, duration


def analyse(rows, t_send, q0, goal_q, joints, settle=0.6):
    before = [r for r in rows if t_send - 0.5 <= r[0] < t_send]
    after = [r for r in rows if t_send <= r[0] < t_send + settle]
    out = []
    for j in range(7):
        direction = math.copysign(1.0, goal_q[j] - q0[j])
        moving = abs(goal_q[j] - q0[j]) > 1e-3
        base = sum(r[1][j] for r in before) / max(1, len(before))
        refs = [r[3][j] for r in before if r[3] is not None]
        sag = (sum(refs) / len(refs) - base) if refs else math.nan
        if moving:
            worst = min((r[1][j] - base) * direction for r in after)
            drop = -min(0.0, worst)
        else:
            drop = max(abs(r[1][j] - base) for r in after)
        eff = sum(r[2][j] for r in before) / max(1, len(before))
        out.append((joints[j], base, sag, drop, eff))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left")
    ap.add_argument("--speed", type=float, default=0.4, help="rad/s")
    ap.add_argument("--hold", type=float, default=2.0,
                    help="seconds holding between moves")
    ap.add_argument("--csv", default="")
    ap.add_argument("--poses", default="",
                    help="poses separated by ';', 7 comma-separated joints each")
    args = ap.parse_args()

    rclpy.init()
    node = Probe(args.side)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    if not node.client.wait_for_server(timeout_sec=5.0):
        raise SystemExit("follow_joint_trajectory action not available")
    while node.latest() is None:
        time.sleep(0.05)

    poses = DEFAULT_POSES
    if args.poses:
        poses = [[float(v) for v in p.split(",")] for p in args.poses.split(";")]
    for k, pose in enumerate(poses):
        time.sleep(args.hold)
        t_send, q0, dur = node.move(pose, args.speed)
        time.sleep(0.2)
        with node.lock:
            rows = list(node.rows)
        print(f"\n== move {k}: {dur:.1f}s to {pose}")
        print(f"{'joint':22s} {'q_before':>9s} {'sag_rest':>9s} "
              f"{'drop':>8s} {'effort':>7s}   (deg, deg, deg, Nm)")
        for name, base, sag, drop, eff in analyse(rows, t_send, q0, pose,
                                                  node.joints):
            print(f"{name:22s} {math.degrees(base):9.2f} "
                  f"{math.degrees(sag):9.2f} {math.degrees(drop):8.2f} "
                  f"{eff:7.2f}")
    time.sleep(args.hold)

    if args.csv:
        with node.lock:
            rows = list(node.rows)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + [f"q{i}" for i in range(1, 8)]
                       + [f"eff{i}" for i in range(1, 8)]
                       + [f"ref{i}" for i in range(1, 8)])
            for t, q, e, r in rows:
                w.writerow([t] + q + e + (r if r else [math.nan] * 7))
        print(f"\nsaved {len(rows)} samples to {args.csv}")
    executor.shutdown()
    spin.join()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
