// Copyright 2025 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Geometry>
#include <string>
#include <vector>

namespace openarm_hardware {

// Static gravity torques for one arm, computed from the URDF inertials.
//
// The MIT loop in the motor is a pure PD: tau = kp*(q_ref - q) + kd*(dq_ref -
// dq) + tau_ff. Without tau_ff the arm settles below the reference by
// tau_gravity/kp, and every new trajectory that starts from the measured state
// (MoveIt always does) moves the reference down to that drooped position, so
// the arm drops once more before it moves. Feeding these torques as tau_ff
// removes the steady-state error.
class GravityModel {
 public:
  // Builds the model for joints openarm_{prefix}joint1..N, including every
  // link below joint1 (hand and fingers too). Gravity is -Z of the URDF root.
  bool init(const std::string& urdf_xml, const std::string& arm_prefix,
            size_t arm_dof, std::string& error);

  // Torque each arm joint must supply to hold the arm still at q.
  void compute(const std::vector<double>& q, std::vector<double>& tau) const;

 private:
  struct Node {
    int parent = -1;     // index in nodes_, -1 = fixed base of the arm
    int subtree_end = 0;  // nodes [this, subtree_end) are below this joint
    int q_index = -1;     // arm joint index, -1 for joints held at zero
    bool prismatic = false;
    Eigen::Isometry3d origin = Eigen::Isometry3d::Identity();
    Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
    double mass = 0.0;
    Eigen::Vector3d com = Eigen::Vector3d::Zero();  // in child link frame
  };

  Eigen::Isometry3d base_ = Eigen::Isometry3d::Identity();
  Eigen::Vector3d gravity_{0.0, 0.0, -9.81};
  std::vector<Node> nodes_;
  std::vector<int> arm_nodes_;
};

}  // namespace openarm_hardware
