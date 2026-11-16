FROM ubuntu:26.04

ENV DEBIAN_FRONTEND="noninteractive" TERM="xterm"
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# dependencies
RUN apt update && apt install -q -y --no-install-recommends \
      wget \
      curl \
      build-essential \
      git git-lfs \
      sudo \
      locate \
      python3-sympy \
      libglfw3-dev \
      python3-dev \
      python3-pip \
      python3-numpy \
      && apt clean \
      && rm -rf /var/lib/apt/lists/*

# Utilities
RUN apt update && apt install -q -y --no-install-recommends \
      cmake \
      lsb-release \
      # Lanelet2 dependencies (Additional needs: libeigen3-dev & python-catkin-tools)
      libboost-dev \
      libboost-all-dev \
      libboost-all-dev \
      libpugixml-dev \
      libboost-python-dev \
      libqt5svg5-dev qttools5-dev \
      libassimp-dev \
      # ros_template
      # other stuff
      libopenblas-dev \
      autotools-dev \
      automake \
      libgoogle-glog-dev \
      googletest \
      libgtest-dev \
      && apt clean \
      && rm -rf /var/lib/apt/lists/*

# install comfort programms
RUN apt update && apt install -q -y --no-install-recommends \
      meld \
      vim \
      # fuse to make appimages work inside docker
      fuse \
      # libgsl for imu_processing
      libgsl-dev \
      # distcc
      distcc distccmon-gnome ccache netcat-openbsd \
      # trajectory_prediction dependencies
      libglm-dev \
      # Grid map matlab exporter, gnuplot for clustering_lib
      libmatio-dev libhdf5-dev gnuplot-nox \
      # Install opencv here, so that we have all necessary dependencies and can overwrite the cv2 python module later
      libopencv-dev python3-opencv \
      # other dependencies
      gnupg \
      software-properties-common \
      && apt clean \
      && rm -rf /var/lib/apt/lists/*

# Install ninja 1.11.0 (for --quiet option, rather old but works well)
RUN cd /tmp && \
    git clone --branch stderr https://github.com/dendy/ninja.git && \
    cd ninja && \
    cmake -Bbuild -DCMAKE_INSTALL_PREFIX=/opt/ninja && \
    cmake --build build --parallel $(nproc --all) && \
    cd build && \
    make install && \
    cd /tmp && \
    rm -rf ninja
ENV PATH=/opt/ninja/bin:$PATH

# python utilities
RUN pip3 install --upgrade --no-cache-dir --break-system-packages \
      scikit-learn scikit-image pytest

# Remove cmake python package which provides a newer cmake version
RUN pip3 uninstall -y --break-system-packages cmake

# git_hooks: clang-17, some python dependencies
RUN apt update \
    && apt install -q -y --no-install-recommends clang-format-17 clang-tidy-17 clang-tools-17 clang-17 libclang-common-17-dev clangd-17 lld-17 llvm-17 libc++-17-dev libc++abi-17-dev libomp-17-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Overwrite googletest installation because of a bug in GCC <= 11 (see https://github.com/google/googletest/pull/3993)
RUN cd /tmp/ && git clone https://github.com/google/googletest.git && mkdir -p googletest/build && cd googletest/build && cmake .. && make install

# add ros2 repo
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg

# add ros2 to apt sources
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main"  > /etc/apt/sources.list.d/ros2.list

# install deps
RUN apt update && apt install -q -y --no-install-recommends \
      python3-colcon-common-extensions \
      ros-dev-tools \
    && pip3 install --upgrade --no-cache-dir --break-system-packages \
      flake8-blind-except \
      flake8-builtins \
      flake8-class-newline \
      flake8-comprehensions \
      flake8-deprecated \
      flake8-docstrings \
      flake8-import-order \
      flake8-quotes \
      pytest-repeat \
      pytest-rerunfailures \
      pytest \
      setuptools

# install deps
RUN apt update && apt install -q -y --no-install-recommends \
      python3-flake8 \
      python3-pip \
      python3-pytest-cov \
      python3-setuptools \
    && pip3 install --upgrade --no-cache-dir --break-system-packages \
      flake8-blind-except \
      flake8-builtins \
      flake8-class-newline \
      flake8-comprehensions \
      flake8-deprecated \
      flake8-docstrings \
      flake8-import-order \
      flake8-quotes \
      pytest-repeat \
      pytest-rerunfailures \
      pytest \
      setuptools


# other pip packages:
#  - python-can, cantools: for peakcan_to_topic_node
#  - grpcio-tools: for ROS multimaster build
#  - ffmpeg-python: for aduulm_tools
#  - numcodecs, blosc: for amqp_topic_transceiver
#  - bson gevent: for ICT4CART v2x
#  - neovim-remote, neovim: for neovim users
#  - utm, tqdm: utilities
# removal of selected system install python packages is necessary to install newer versions via pip
RUN pip3 install --upgrade --no-cache-dir --break-system-packages  \
      autopep8 pylint matplotlib requests python-gitlab stonesoup \
     tikzplotlib \
      utm tqdm "moderngl<6.0.0" && \
    # pillow with SIMD support
    pip3 uninstall -y --break-system-packages pillow && \
    CC="cc -mavx2" pip3 install --upgrade --no-cache-dir --force-reinstall --break-system-packages pillow-simd

# jsonschema networkx deepdiff: Required by orchestrator
RUN pip3 install --upgrade --break-system-packages \
    nanobind \
    parameterized \
    reportlab \
    shapely \
    scikit-image \
    deepdiff \
    mcap-ros2-support \
    gcovr

# generate locales
RUN apt -y update && apt install -y locales && apt clean && rm -rf /var/lib/apt/lists/* && echo "en_US.UTF-8 UTF-8" | tee -a /etc/locale.gen && locale-gen

# rsync
RUN apt -y update && apt install -y rsync && apt clean && rm -rf /var/lib/apt/lists/*

# Install newer version of transforms3d, because of numpy issues in the version available for ubuntu 22.04
RUN pip3 install --no-cache-dir --break-system-packages transforms3d

# figcone for tracking
RUN cd /tmp && \
    git clone https://github.com/kamchatka-volcano/figcone.git && \
    mkdir -p figcone/build && \
    cd figcone/build && \
    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DFIGCONE_USE_YAML=ON .. && \
    make install && \
    cd /tmp/ && \
    rm -rf figcone

# type_safe for tracking
RUN cd /tmp && \
    git clone https://github.com/foonathan/type_safe && \
    mkdir -p type_safe/build && \
    cd type_safe/build && \
    cmake .. && \
    make install && \
    cd /tmp/ && \
    rm -rf type_safe

# debug_assert needed by type_safe for tracking
RUN cd /tmp && \
    git clone https://github.com/foonathan/debug_assert.git && \
    mkdir -p debug_assert/build && \
    cd debug_assert/build && \
    cmake .. && \
    make install && \
    cd /tmp/ && \
    rm -rf debug_assert

RUN apt -y update && apt install -y libfreetype-dev

# imgui + implot (e.g. used by trajectory planning, gridfusion)
COPY files/* /tmp/
RUN cd /tmp && \
    git clone --branch v1.90.5-docking https://github.com/ocornut/imgui.git && \
    cp /tmp/CMakeLists.imgui.txt /tmp/imgui/CMakeLists.txt && \
    cp /tmp/imgui.cmake.in /tmp/imgui/imgui.cmake.in && \
    mkdir -p /usr/share/fonts/imgui && \
    cp /tmp/imgui/misc/fonts/*.ttf /usr/share/fonts/imgui && \
    mkdir -p imgui/build && \
    cd imgui/build && \
    cmake .. && \
    make install && \
    cd /tmp/ && \
    rm -rf imgui

RUN cd /tmp && \
    git clone --branch v0.16 https://github.com/epezent/implot.git && \
    cp /tmp/CMakeLists.implot.txt /tmp/implot/CMakeLists.txt && \
    mkdir -p implot/build && \
    cd implot/build && \
    cmake .. && \
    make install && \
    cd /tmp/ && \
    rm -rf implot


RUN apt -y update && apt -y install \
    ros-lyrical-ament-cmake  \
    ros-lyrical-ament-cmake-python  \
    ros-lyrical-ament-cmake-core \
    ros-lyrical-ament-cmake-libraries \
    ros-lyrical-ament-package \
    ros-lyrical-ament-cmake-gtest

RUN apt -y update && apt -y install libeigen3-dev


RUN apt -y update && apt -y install librsvg2-2

RUN mkdir /workspace && \
    cd /workspace && \
    git clone https://github.com/uulm-mrm/aduulm_cmake_tools.git && \
    cd aduulm_cmake_tools && \
    git submodule update --init --recursive && \
    cd .. && \
    git clone https://github.com/uulm-mrm/aduulm_logger.git && \
    cd aduulm_logger && \
    git submodule update --init --recursive && \
    sed -i 's/VERSION 3\.3 FATAL_ERROR/VERSION 3.10 FATAL_ERROR/' library/CMakeLists.txt && \
    cd .. && \
    git clone https://github.com/uulm-mrm/minimal_latency_in_sequence_buffer.git && \
    cd minimal_latency_in_sequence_buffer && \
    git submodule update --init --recursive && \
    cd .. && \
    git clone https://github.com/uulm-mrm/eslimpp.git && \
    cd eslimpp && \
    git submodule update --init --recursive

RUN apt -y update && apt -y install libdbus-1-dev libfreetype-dev libtbb-dev libglfw3-dev build-essential

# tracy profiler
RUN apt -y update && apt -y install libdbus-1-dev libfreetype-dev libtbb-dev
RUN apt -y update && apt -y install --no-install-recommends \
    libdbus-1-dev libfreetype-dev libtbb-dev \
    libwayland-dev wayland-protocols libxkbcommon-dev \
    libegl1-mesa-dev libgl1-mesa-dev pkg-config \
    && apt clean && rm -rf /var/lib/apt/lists/*
RUN apt -y update && apt -y install --no-install-recommends \
 libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libx11-dev libxext-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
    git clone https://github.com/wolfpld/tracy.git --branch master && \
    cd tracy && \
    mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DTRACY_STATIC=OFF -DTRACY_ONLY_LOCALHOST=ON .. && cmake --build . -j && make install && \
    cd ../profiler && \
    mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DLEGACY=ON .. && cmake --build . -j && \
    mv tracy-profiler /usr/bin/ && \
    cd /tmp/ && \
    rm -rf tracy
    
RUN apt-get update && apt-get install -q -y --no-install-recommends \
      libjemalloc-dev \
      mold \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*
      
# Gcc family compilers
ARG GCC_VERSION=14
RUN apt-get update && apt-get install -q -y --no-install-recommends \
        gcc-$GCC_VERSION \
        g++-$GCC_VERSION \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$GCC_VERSION 10 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$GCC_VERSION 10 \
    && update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-$GCC_VERSION 10
