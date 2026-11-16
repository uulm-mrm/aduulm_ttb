#pragma once
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif

#include <hpx/hpx.hpp>
#include "hpx/synchronization/mutex.hpp"
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/hpx_start.hpp>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
