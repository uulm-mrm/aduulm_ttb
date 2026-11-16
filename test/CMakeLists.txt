find_package(GTest REQUIRED)

if (NOT ${GTEST_FOUND})
    message(WARNING "GTest not found, skipping tests for tracking_lib")
    return()
endif ()


enable_testing()

function(create_test test_name exec_names)
    add_executable(${test_name}
            ${exec_names}
            logger_setup.cpp
            test_main.cpp
    )
    target_link_libraries(${test_name}
            PUBLIC
            tracking_lib::tracking_lib
            GTest::gtest_main
            HPX::hpx
            XAD::xad
    )
    target_compile_options(${test_name}
            PRIVATE
            -Wno-unused-variable
    )
    target_compile_features(${test_name} PUBLIC cxx_std_23)


    # register the test with CTest, so we execute it automatically when running CTest later
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    #    gtest_discover_tests(${test_name} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    # Set target build directory
    set_target_properties(${test_name}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
endfunction()

set(all_test_files
        DefaultTrackingSimulation/test_launch_config.cpp
        ##################################################
        AzimuthOnly/test_azimuth_only.cpp
        ##################################################
        Distributions/test_Distributions.cpp
        ##################################################
        MeasurementModels/test_GenericBoxModel.cpp
        MeasurementModels/test_SensorInformation.cpp
        MeasurementModels/test_cv_with_xy.cpp
        MeasurementModels/test_ctrv_box_with_box.cpp
        MeasurementModels/test_ctrv_box_with_camera.cpp
        ################################################
        # StateModels/test_stateModels.cpp
        StateModels/test_cv.cpp
        StateModels/test_ctrv.cpp
        StateModels/test_ctrv_cv.cpp
        ###################################################
        misc/test_unscentedUpdate.cpp
        misc/test_syncro.cpp
        misc/test_boost_multi_index.cpp
        misc/test_fold.cpp
        misc/test_state.cpp
        misc/test_components.cpp
        misc/test_HPX.cpp
        misc/test_min_max_hull.cpp
        ##################################################
        Graph/test_DiGraphImpl.cpp
        Graph/test_DiGraph.cpp
        #####################################################
        LMB_GLMB_Update/test_lmb_mo_update.cpp
        #####################################################
        Transformations/test_transformations.cpp
        ######################################################
        TTBHelpers/test_murty.cpp
        TTBHelpers/test_gibbs.cpp
        #########################################################
        Orientation/orientation_test.cpp
        Orientation/orientation_update_test.cpp
        Orientation/orientation_update_test_eastwards.cpp
)

create_test(Test
        "${all_test_files}"
)
