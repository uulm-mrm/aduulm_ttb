#####################
## Python Bindings ##
#####################

find_package(Eigen3 REQUIRED)
find_package(aduulm_logger_lib REQUIRED)
find_package(minimal_latency_buffer REQUIRED)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Detect the installed nanobind package and import it into CMake
execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import nanobind; import pathlib; print(pathlib.Path(nanobind.cmake_dir()).parent)"
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG)

set(TARGET _${PROJECT_NAME}_python_api)

if (nanobind_FOUND)
    nanobind_add_module(${TARGET}
            PythonBindings.cpp
    )
    target_compile_features(${TARGET}
            PUBLIC
            cxx_std_23
    )
    target_compile_options(nanobind-static
            PRIVATE
            -Wno-shadow
            -Wno-pedantic
    )
    target_compile_options(${TARGET}
            PRIVATE
            -Wno-shadow
            -Wno-pedantic
    )
    target_link_libraries(${TARGET}
            PUBLIC
            tracking_lib::tracking_lib
    )
    install(TARGETS ${TARGET}
            LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME} COMPONENT Runtime
    )

else ()
    message(WARNING "Nanobind not found, skip building python bindings for tracking_lib")
endif ()
