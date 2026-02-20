cmake_minimum_required(VERSION 2.8.3)

project(Mobile_Leveling_Platform_V2)

find_package(catkin REQUIRED   
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  mavros_msgs
  drl_uav
  tf
)
find_package(cmake_modules REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

add_message_files(FILES SensorState.msg joint_vel.msg kalman.msg)

generate_messages(DEPENDENCIES std_msgs sensor_msgs geometry_msgs)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs tf
  DEPENDS Eigen
)

find_package(roslaunch)

foreach(dir config launch meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
	include_directories(include ${catkin_INCLUDE_DIRS})
	add_executable(cont src/cont.cpp)
	target_link_libraries(cont ${catkin_LIBRARIES})
	add_dependencies(cont ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

	add_executable(srvcommand src/srvcommand.cpp)
	target_link_libraries(srvcommand ${catkin_LIBRARIES} ${Eigen_LIBRARIES})
	add_dependencies(srvcommand ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

	
