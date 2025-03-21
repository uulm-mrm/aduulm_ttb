find_package(GTest REQUIRED)

if (NOT ${GTEST_FOUND})
    message(WARNING "GTest not found, skipping tests for tracking_lib")
    return()
endif ()
enable_testing()

# executable
SET(TEST_NAME tracking_lib_test)

# Add source files
add_executable(${TEST_NAME}
        DefaultTrackingSimulation/test_launch_config.cpp
        ########################################################
        AzimuthOnly/test_azimuth_only.cpp
        ########################################################
        Distributions/test_Distributions.cpp
        ########################################################
        MeasurementModels/test_GenericBoxModel.cpp
        MeasurementModels/test_SensorInforamtion.cpp
        MeasurementModels/test_cv_with_xy.cpp
        MeasurementModels/test_ctrv_box_with_box.cpp
        MeasurementModels/test_ctrv_box_with_camera.cpp
        ######################################################
        StateModels/test_stateModels.cpp
        StateModels/test_cv.cpp
        StateModels/test_ctrv.cpp
        StateModels/test_ctrv_cv.cpp
        #########################################################
        misc/test_unscentedUpdate.cpp
        #########################################################
        Graph/test_DiGraphImpl.cpp
        Graph/test_DiGraph.cpp
        #########################################################
        LMB_GLMB_Update/test_lmb_mo_update.cpp
        #########################################################
        MultiObjectDistributions/test_GLMB_generateHypotheses.cpp
        #########################################################
        Transformations/test_transformations.cpp
        #########################################################
        TTBHelpers/test_murty.cpp
        TTBHelpers/test_gibbs.cpp
        #########################################################
        logger_setup.cpp
)

#MESSAGE("CMAKE Source DIR ${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME lib_test COMMAND ${TEST_NAME} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# Link libraries
target_link_libraries(${TEST_NAME}
        PUBLIC
        tracking_lib::tracking_lib
        GTest::gtest_main
)

# Set target build directory
set_target_properties(${TEST_NAME}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)

# coverage
if (${COVERAGE})
    message(STATUS "Building unit tests for coverage analysis")
    target_link_libraries(${TEST_NAME} PRIVATE -fprofile-arcs -ftest-coverage)
    target_compile_options(${TEST_NAME} PRIVATE --coverage -g -O0)
endif ()
