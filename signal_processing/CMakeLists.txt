cmake_minimum_required(VERSION 3.8)
project(signal_processing)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(ament_cmake_pytest REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/dead_zone.cpp
  src/low_pass_filter.cpp
  src/one_euro_filter.cpp
  src/saturation.cpp
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen
)

ament_python_install_package(${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  ament_add_gtest(test_signal_processing_cpp
    test/test_signal_processing.cpp
  )
  target_link_libraries(test_signal_processing_cpp
    ${PROJECT_NAME}
    Eigen3::Eigen
  )

  ament_add_pytest_test(test_signal_processing_py
    test/test_signal_processing.py
    PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}"
  )
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)

ament_package()
