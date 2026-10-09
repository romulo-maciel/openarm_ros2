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

#include "openarm_hardware/gravity_model.hpp"

#include <urdf/model.h>

namespace openarm_hardware {

namespace {

Eigen::Isometry3d to_eigen(const urdf::Pose& p) {
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.translation() << p.position.x, p.position.y, p.position.z;
  t.linear() = Eigen::Quaterniond(p.rotation.w, p.rotation.x, p.rotation.y,
                                  p.rotation.z)
                   .toRotationMatrix();
  return t;
}

}  // namespace

bool GravityModel::init(const std::string& urdf_xml,
                        const std::string& arm_prefix, size_t arm_dof,
                        std::string& error) {
  urdf::Model model;
  if (!model.initString(urdf_xml)) {
    error = "failed to parse URDF";
    return false;
  }

  const std::string first = "openarm_" + arm_prefix + "joint1";
  auto first_joint = model.getJoint(first);
  if (!first_joint) {
    error = "joint " + first + " not in URDF";
    return false;
  }

  // Fixed transform root -> parent link of joint1.
  base_ = Eigen::Isometry3d::Identity();
  for (auto link = model.getLink(first_joint->parent_link_name);
       link && link->parent_joint;) {
    auto j = link->parent_joint;
    if (j->type != urdf::Joint::FIXED) {
      error = "joint " + j->name + " above " + first + " is not fixed";
      return false;
    }
    base_ = to_eigen(j->parent_to_joint_origin_transform) * base_;
    link = model.getLink(j->parent_link_name);
  }

  nodes_.clear();
  arm_nodes_.assign(arm_dof, -1);

  // Depth-first so that every subtree is a contiguous range of nodes_.
  auto add = [&](auto&& self, const urdf::JointConstSharedPtr& j,
                 int parent) -> void {
    const int idx = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    Node& n = nodes_.back();
    n.parent = parent;
    n.origin = to_eigen(j->parent_to_joint_origin_transform);
    n.axis << j->axis.x, j->axis.y, j->axis.z;
    n.prismatic = j->type == urdf::Joint::PRISMATIC;
    for (size_t i = 0; i < arm_dof; ++i) {
      if (j->name == "openarm_" + arm_prefix + "joint" + std::to_string(i + 1)) {
        n.q_index = static_cast<int>(i);
        arm_nodes_[i] = idx;
      }
    }
    auto child = model.getLink(j->child_link_name);
    if (child && child->inertial) {
      n.mass = child->inertial->mass;
      n.com = to_eigen(child->inertial->origin).translation();
    }
    if (child) {
      for (const auto& cj : child->child_joints) self(self, cj, idx);
    }
    nodes_[idx].subtree_end = static_cast<int>(nodes_.size());
  };
  add(add, first_joint, -1);

  for (size_t i = 0; i < arm_dof; ++i) {
    if (arm_nodes_[i] < 0) {
      error = "joint openarm_" + arm_prefix + "joint" + std::to_string(i + 1) +
              " is not below " + first;
      return false;
    }
  }
  return true;
}

void GravityModel::compute(const std::vector<double>& q,
                           std::vector<double>& tau) const {
  const size_t n = nodes_.size();
  std::vector<Eigen::Isometry3d> child_tf(n);
  std::vector<Eigen::Vector3d> axis_w(n), joint_w(n), weight_at(n), force(n);

  for (size_t k = 0; k < n; ++k) {
    const Node& nd = nodes_[k];
    const Eigen::Isometry3d joint_tf =
        (nd.parent < 0 ? base_ : child_tf[nd.parent]) * nd.origin;
    const double qk = nd.q_index >= 0 ? q[nd.q_index] : 0.0;
    Eigen::Isometry3d motion = Eigen::Isometry3d::Identity();
    if (nd.prismatic) {
      motion.translation() = nd.axis * qk;
    } else {
      motion.linear() = Eigen::AngleAxisd(qk, nd.axis).toRotationMatrix();
    }
    child_tf[k] = joint_tf * motion;
    axis_w[k] = joint_tf.linear() * nd.axis;
    joint_w[k] = joint_tf.translation();
    weight_at[k] = child_tf[k] * nd.com;
    force[k] = nd.mass * gravity_;
  }

  tau.assign(arm_nodes_.size(), 0.0);
  for (size_t i = 0; i < arm_nodes_.size(); ++i) {
    const Node& ji = nodes_[arm_nodes_[i]];
    double t = 0.0;
    for (int k = arm_nodes_[i]; k < ji.subtree_end; ++k) {
      t += axis_w[arm_nodes_[i]].dot(
          (weight_at[k] - joint_w[arm_nodes_[i]]).cross(force[k]));
    }
    tau[i] = -t;  // the motor has to cancel the gravity moment
  }
}

}  // namespace openarm_hardware
