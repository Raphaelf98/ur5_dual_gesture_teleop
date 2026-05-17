search_mode=OPTIMIZE_MAX_JOINT
srdf_filename=dual_ur5.srdf
robot_name_in_srdf=dual_ur5
moveit_config_pkg=dual_ur5_moveit_config
robot_name=dual_ur5
planning_group_name=right_arm
ikfast_plugin_pkg=dual_ur5_right_arm_ikfast_plugin
base_link_name=right_base_link
eef_link_name=right_tool0
ikfast_output_path=/home/raphael/robolab/ros2_ws/src/dual_ur5_right_arm_ikfast_plugin/src/dual_ur5_right_arm_ikfast_solver.cpp

rosrun moveit_kinematics create_ikfast_moveit_plugin.py\
  --search_mode=$search_mode\
  --srdf_filename=$srdf_filename\
  --robot_name_in_srdf=$robot_name_in_srdf\
  --moveit_config_pkg=$moveit_config_pkg\
  $robot_name\
  $planning_group_name\
  $ikfast_plugin_pkg\
  $base_link_name\
  $eef_link_name\
  $ikfast_output_path
