/* 
 * Night-Voyager: Consistent and Efficient Nocturnal Vision-Aided State Estimation in Object Maps
 * Copyright (C) 2025 Night-Voyager Contributors
 * 
 * For technical issues and support, please contact Tianxiao Gao at <ga0.tianxiao@connect.um.edu.mo>
 * or Mingle Zhao at <zhao.mingle@connect.um.edu.mo>. For commercial use, please contact Prof. Hui Kong at <huikong@um.edu.mo>.
 * 
 * This file is subject to the terms and conditions outlined in the 'LICENSE' file,
 * which is included as part of this source code package.
 */
#include "core/NightVoyagerOptions.h"
#include "msckf_iekf/VILManager.h"
#include "prior_pose/PriorPoseManager.h"
#include "streetlight_matcher/PcdManager.h"
#include "utils/Print.h"
#include "visualizer/Visualizer.h"
#include <iostream>
#include <ros/ros.h>
#include <thread>

using namespace night_voyager;

int main(int argc, char **argv) {
    ros::init(argc, argv, "Night_Voyager");
    ros::NodeHandle nh;

    NightVoyagerOptions options;
    options.root_dir = ROOT_DIR;
    options.param_load(nh);
    // Verbosity
    std::string verbosity = "INFO";
    Printer::setPrintLevel(verbosity);

    auto pcd_manager = std::make_shared<PcdManager>(options);
    auto prior_pose_manager = std::make_shared<PriorPoseManager>(options);
    auto sys = std::make_shared<VILManager>(options, pcd_manager, prior_pose_manager);
    auto viz = std::make_shared<Visualizer>(nh, options, sys);

    ros::AsyncSpinner spinner(0);
    spinner.start();

    while (ros::ok()) {
        viz->run();
    }
    ros::waitForShutdown();
    ros::shutdown();

    return EXIT_SUCCESS;
}
