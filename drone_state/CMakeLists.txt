cmake_minimum_required(VERSION 3.0.2)
project(drone_state)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  mavros
  roscpp
  std_msgs
  sensor_msgs
  aruco_msgs
  drl_uav
  message_generation
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
 add_message_files(
   FILES
   variance.msg
   custom_gain.msg
 )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   getEnvinfo.srv
#   getRLflag.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
 generate_messages(
  DEPENDENCIES
   sensor_msgs
   std_msgs
   drl_uav
 )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES drone_state
  CATKIN_DEPENDS mavros roscpp sensor_msgs Mobile_Leveling_Platform_V2 drl_uav
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/drone_state.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/drone_state_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_drone_state.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(drone_state src/drone_control/drone_state.cpp)
target_link_libraries(drone_state ${catkin_LIBRARIES})

add_executable(take_off_and_land src/drone_control/take_off_and_land.cpp)
target_link_libraries(take_off_and_land ${catkin_LIBRARIES})

add_executable(offb_node src/drone_control/offb_node.cpp)
target_link_libraries(offb_node ${catkin_LIBRARIES})

add_executable(gps_vision_tracking src/drone_control/gps_vision_tracking.cpp)
target_link_libraries(gps_vision_tracking ${catkin_LIBRARIES})

add_executable(RL_quad_env src/drone_control/RL_quad_env.cpp)
target_link_libraries(RL_quad_env ${catkin_LIBRARIES})

add_executable(RL_quad_env_save src/drone_control/RL_quad_env_save.cpp)
target_link_libraries(RL_quad_env_save ${catkin_LIBRARIES})

add_executable(RL_quad_env_save_ddpg_sepmem src/drone_control/RL_quad_env_save_ddpg_sepmem.cpp)
target_link_libraries(RL_quad_env_save_ddpg_sepmem ${catkin_LIBRARIES})

add_executable(RL_quad_env_save_normal src/drone_control/RL_quad_env_save_normal.cpp)
target_link_libraries(RL_quad_env_save_normal ${catkin_LIBRARIES})

add_executable(RL_quad_env_save_per src/drone_control/RL_quad_env_save_per.cpp)
target_link_libraries(RL_quad_env_save_per ${catkin_LIBRARIES})

add_executable(RL_quad_env_save_per_fix1 src/drone_control/RL_quad_env_save_per_fix1.cpp)
target_link_libraries(RL_quad_env_save_per_fix1 ${catkin_LIBRARIES})

add_executable(RL_quad_env_save_normal_fix1 src/drone_control/RL_quad_env_save_normal_fix1.cpp)
target_link_libraries(RL_quad_env_save_normal_fix1 ${catkin_LIBRARIES})

add_executable(RL_quad_env_save_ddpg_sepmem_fix1 src/drone_control/RL_quad_env_save_ddpg_sepmem_fix1.cpp)
target_link_libraries(RL_quad_env_save_ddpg_sepmem_fix1 ${catkin_LIBRARIES})

add_executable(test_gpsvision.cpp src/drone_control/test_gpsvision.cpp)
target_link_libraries(test_gpsvision.cpp ${catkin_LIBRARIES})

add_executable(circle src/drone_control/circle.cpp)
target_link_libraries(circle ${catkin_LIBRARIES})

add_executable(keyboard src/drone_control/keyboard_cmd.cpp)
target_link_libraries(keyboard ${catkin_LIBRARIES})

