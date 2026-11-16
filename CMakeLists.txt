cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(tracking_lib)

find_package(aduulm_cmake_tools)
if (NOT ${aduulm_cmake_tools_FOUND})
    MESSAGE(FATAL_ERROR "aduulm_cmake_tools not found. Clone this repo into the same level as the tracking repo.")
endif ()
find_package(aduulm_logger_lib)
if (NOT ${aduulm_logger_lib_FOUND})
    MESSAGE(FATAL_ERROR "aduulm_logger_lib not found. Clone this repo into the same level as the tracking repo.")
endif ()
find_package(minimal_latency_buffer)
if (NOT ${minimal_latency_buffer_FOUND})
    MESSAGE(FATAL_ERROR "minimal_latency_buffer not found. Clone this repo into the same level as the tracking repo.")
endif ()
find_package(subjective_logic_lib)

get_version_string_from_package_xml(.)
show_build_info("library")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif ()
# Dependencies included in docker
find_package(Eigen3 REQUIRED)
add_compile_definitions(EIGEN_DONT_PARALLELIZE)
add_compile_definitions(EIGEN_MAX_ALIGN_BYTES=0)
add_compile_definitions(EIGEN_MAX_STATIC_ALIGN_BYTES=0)
add_compile_definitions(EIGEN_MALLOC_ALREADY_ALIGNED=0)

find_package(Boost REQUIRED)
find_package(figcone REQUIRED)
find_package(Tracy REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(imgui REQUIRED)
find_package(implot REQUIRED)
find_package(OpenGL REQUIRED)

if (TRACY_ENABLE MATCHES ON)
    MESSAGE("Tracy Enabled")
else ()
    MESSAGE("Tracy Disabled")
endif ()

set(HPX_WITH_STACKTRACES ON CACHE BOOL "Disable HPX stacktraces")
set(HPX_WITH_TESTS OFF CACHE BOOL "" FORCE)
set(HPX_WITH_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HPX_WITH_DOCUMENTATION OFF CACHE BOOL "" FORCE)
set(HPX_WITH_NETWORKING OFF CACHE BOOL "" FORCE)
set(HPX_WITH_MALLOC "jemalloc" CACHE STRING "" FORCE)
set(HPX_WITH_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(HPX_TRACY_WITH_TRACY OFF CACHE BOOL "" FORCE)
set(HPX_WITH_FETCH_ASIO ON CACHE BOOL "" FORCE)

add_subdirectory(third_party/hpx SYSTEM)

add_subdirectory(third_party/xad SYSTEM)
find_package(xad)

set(sources
        src/BirthModels/BaseBirthModel.cpp
        src/BirthModels/DynamicBirthModel.cpp
        src/BirthModels/StaticBirthModel.cpp
        src/BirthModels/ROIBirthModel.cpp
        ################################################################################################################
        src/Classification/MeasClassification.cpp
        src/Classification/StateClassification.cpp
        ################################################################################################################
        src/DataContainer/Database.cpp
        ################################################################################################################
        src/DataContainer/DataSourceContainer.cpp
        ################################################################################################################
        src/Distributions/PoissonDistribution.cpp
        src/Distributions/GammaDistribution.cpp
        src/Distributions/DirichletDistribution.cpp
        ################################################################################################################
        src/Trackers/NO_Tracker.cpp
        src/Trackers/NN_Tracker.cpp
        src/Trackers/GNN_Tracker.cpp
        src/Trackers/Id_Tracker.cpp
        src/Trackers/Tracker.cpp
        src/Trackers/Nothing_Tracker.cpp
        ################################################################################################################
        src/Graph/GraphImpl.cpp
        src/Graph/GeneralizedKBestSelectionAlgorithm.cpp
        ################################################################################################################
        src/Map/Map.cpp
        ################################################################################################################
        src/MeasurementModels/BaseMeasurementModel.cpp
        src/MeasurementModels/GaussianMeasurementModel.cpp
        ################################################################################################################
        src/Measurements/SensorInformation.cpp
        ################################################################################################################
        src/Misc/Numeric.cpp
        src/Misc/Assignments.cpp
        src/Misc/AngleNormalization.cpp
        src/Misc/GibbsSampler.cpp
        src/Misc/Ospa.cpp
        src/Misc/Grouping.cpp
        src/Misc/HungarianMethod.cpp
        src/Misc/MurtyAlgorithm.cpp
        src/Misc/SeparatingAxisTest.cpp
        src/Misc/Profiler.cpp
        src/Misc/ProportionalAllocation.cpp
        src/Misc/logger_setup.cpp
        src/Misc/Region.cpp
        ################################################################################################################
        src/MultiObjectStateDistributions/GLMBDistribution.cpp
        src/MultiObjectStateDistributions/Hypothesis.cpp
        src/MultiObjectStateDistributions/LMBDistribution.cpp
        src/MultiObjectStateDistributions/PoissonPointProcess.cpp
        src/MultiObjectStateDistributions/MBMDistribution.cpp
        src/MultiObjectStateDistributions/PMBMDistribution.cpp
        ################################################################################################################
        src/PersistenceModels/ConstantPersistenceModel.cpp
        src/PersistenceModels/ROIPersistenceModel.cpp
        ################################################################################################################
        src/SelfAssessment/SelfAssessment.cpp
        ################################################################################################################
        src/States/EgoMotionDistribution.cpp
        src/States/State.cpp
        ################################################################################################################
        src/StateModels/BaseStateModel.cpp
        src/StateModels/ISCATRStateModel.cpp
        src/StateModels/CAStateModel.cpp
        src/StateModels/CPStateModel.cpp
        src/StateModels/CTRAStateModel.cpp
        src/StateModels/CTRVStateModel.cpp
        src/StateModels/CVStateModel.cpp
        src/StateModels/CTP.cpp
        src/StateModels/PredictionInformation.cpp
        ################################################################################################################
        src/StateTransition/MarkovTransition.cpp
        ################################################################################################################
        src/Transformations/Transformations.cpp
        ################################################################################################################
        src/TTBManager/TTBManager.cpp
        src/TTBManager/Visualization.cpp
        ################################################################################################################
        src/TTBTypes/TTBTypes.cpp
        src/TTBTypes/Components.cpp
        ################################################################################################################
        src/TTTFilters/NoTTTFilter.cpp
        src/TTTFilters/TransTTTFilter.cpp
        src/TTTFilters/TTTHelpers.cpp
)

# TARGETS
set(INCLUDE_INSTALL_DIR include/ CACHE STRING "Path to installed include directory")
set(LIB_INSTALL_DIR lib/ CACHE STRING "Path to installed lib directory")
set(BIN_INSTALL_DIR bin/ CACHE STRING "Path to installed bin directory")
set(SHARE_INSTALL_DIR share/ CACHE STRING "Path to installed share directory")


## build c++ library
add_library(${PROJECT_NAME} SHARED ${sources})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)


if (CMAKE_BUILD_TYPE MATCHES Release)
    MESSAGE("Compile with Release flags")
    target_compile_options(${PROJECT_NAME} PRIVATE
            -O3
            -g3
    )
endif ()
if (CMAKE_BUILD_TYPE MATCHES Debug)
    MESSAGE("Compile with Debug flags")
    target_compile_options(${PROJECT_NAME} PRIVATE
            -O3
            -g3
            -Wall
            -Wextra
            -W
            -pedantic
            -Wnon-virtual-dtor
            -Wduplicated-branches
            -Wpedantic
            -Wduplicated-cond
            -Wswitch#-enum
            -Wcast-qual
            -Wcast-align
            -Wcast-function-type
            -Wunused-function
            -Wtype-limits
            -Wuninitialized
            -Wnonnull
            -Waggressive-loop-optimizations
    )
endif ()

if (UBSAN MATCHES true)
    MESSAGE("Compile with UB sanitizer")
    if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
        MESSAGE(WARNING "Build type not debug")
    endif ()
    target_compile_options(${PROJECT_NAME} PRIVATE
            -O3
            -g3
            -fsanitize=undefined
    )
    target_link_options(${PROJECT_NAME} PUBLIC
            -fsanitize=undefined
    )
endif ()

if (ASAN MATCHES true)
    MESSAGE("Compile with ASAN")
    if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
        MESSAGE(WARNING "Build type not debug")
    endif ()
    target_compile_options(${PROJECT_NAME} PRIVATE
            -O3
            -g3
            -fsanitize=address

    )
    target_link_options(${PROJECT_NAME} PUBLIC
            -fsanitize=address
    )
endif ()
if (TSAN MATCHES true)
    MESSAGE("Compile with TSAN")
    if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
        MESSAGE(WARNING "Build type not debug")
    endif ()
    target_compile_options(${PROJECT_NAME} PRIVATE
            -O3
            -g3
            -fsanitize=thread
    )
    target_link_options(${PROJECT_NAME} PUBLIC
            -fsanitize=thread
    )
endif ()

target_compile_options(${PROJECT_NAME} PRIVATE
        -Wno-empty-body
        -Wno-shadow
        -Wfatal-errors
        -Wno-interference-size
)

target_compile_definitions(${PROJECT_NAME} PUBLIC "-D${package_name}_VERSION=\"${package_version}\"")
target_compile_definitions(${PROJECT_NAME} PRIVATE -DROS_PACKAGE_NAME="${PROJECT_NAME}")

target_include_directories(${PROJECT_NAME}
        SYSTEM INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)
target_include_directories(${PROJECT_NAME}
        PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        Eigen3::Eigen
        aduulm_logger_lib::aduulm_logger_lib
        minimal_latency_buffer::minimal_latency_buffer
        subjective_logic_lib::subjective_logic_lib
        -lstdc++exp
        PRIVATE
        HPX::hpx
        XAD::xad
        figcone::figcone
        Boost::boost
        implot::implot
        imgui::imgui_impl_glfw
        imgui::imgui_impl_opengl3
)



if (TRACY_ENABLE MATCHES ON)
    target_link_libraries(${PROJECT_NAME}
            PRIVATE
            Tracy::TracyClient
    )
endif ()

################
## Installing ##
################
set(PACKAGE_LIBRARY_VERSION "${package_version}")

# Create package version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        VERSION "${PACKAGE_LIBRARY_VERSION}"
        COMPATIBILITY AnyNewerVersion
)

# Install files for all targets
install(TARGETS ${PROJECT_NAME} # (add additional targets here)
        EXPORT ${PROJECT_NAME}Targets # store targets in variable
        INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR}
        LIBRARY DESTINATION ${LIB_INSTALL_DIR} COMPONENT Runtime
        ARCHIVE DESTINATION ${LIB_INSTALL_DIR} COMPONENT Development
        RUNTIME DESTINATION ${BIN_INSTALL_DIR} COMPONENT Runtime
        PUBLIC_HEADER DESTINATION ${INCLUDE_INSTALL_DIR} COMPONENT Development
        BUNDLE DESTINATION ${BIN_INSTALL_DIR} COMPONENT Runtime
)

# Create package config file (to enable find_package())
configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/config.cmake.in" # this file defines the include dirs and libs and may need to be adapted for your package/library
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}
        PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR BIN_INSTALL_DIR
)

if (IS_ROS)
    install(EXPORT ${PROJECT_NAME}Targets
            DESTINATION ${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}
            NAMESPACE ${PROJECT_NAME}::
    )
endif ()

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        DESTINATION ${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${INCLUDE_INSTALL_DIR})

if (NOT IS_ROS_DEVEL)
    INSTALL(FILES ${PROJECT_SOURCE_DIR}/package.xml DESTINATION ${SHARE_INSTALL_DIR}/${PROJECT_NAME})
endif ()

add_subdirectory(test)

setup_git_hooks(CLANG_TIDY_CHECKS "readability-*,bugprone-*,modernize-*,google-*,cppcoreguidelines-*")

add_subdirectory(src/Bindings)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/python/
        DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})

install(PROGRAMS
        ${PROJECT_SOURCE_DIR}/python/tracking_simulator/run_simulation.py
        ${PROJECT_SOURCE_DIR}/python/tracking_simulator/run_data_viz.py
        ${PROJECT_SOURCE_DIR}/python/tracking_simulator/create_scenario.py
        ${PROJECT_SOURCE_DIR}/python/tracking_simulator/gospa.py
        ${PROJECT_SOURCE_DIR}/python/tracking_simulator/eval_nuscenes.py
        ${PROJECT_SOURCE_DIR}/python/tracking_simulator/get_measurements.py
        ${PROJECT_SOURCE_DIR}/python/tracking_simulator/nuScenes_utils.py
        ${PROJECT_SOURCE_DIR}/python/tracking_simulator/read_data_nuscenes.py
        DESTINATION bin
)
