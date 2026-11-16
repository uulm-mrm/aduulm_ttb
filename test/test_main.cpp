#include <gtest/gtest.h>
#include "tracking_lib/Misc/HPXWrapper.h"

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  hpx::init_params params;
  params.cfg = { "hpx.os_threads=4", "hpx.stacks.small_size=0x40000" };
  params.rp_callback = [](hpx::resource::partitioner& rp, hpx::program_options::variables_map const&) {
    // Create a dedicated single-threaded pool for viz
    rp.create_thread_pool("viz_pool", hpx::resource::scheduling_policy::local_priority_fifo);

    // Assign exactly one PU to it (taken from the last core)
    auto& numa_domains = rp.numa_domains();
    auto& last_core = numa_domains.back().cores().back();
    auto& pu = last_core.pus().back();

    rp.add_resource(pu, "viz_pool");  // moves PU out of default pool
  };
  if (!hpx::start(nullptr, argc, argv, params))
  {
    throw std::runtime_error("Failed to start HPX runtime");
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  int const result = hpx::run_as_hpx_thread([&] {
    int const res = RUN_ALL_TESTS();  // Check for lingering work
    return res;
  });
  std::cout << "Finished tests" << std::endl;
  std::_Exit(result);
}
