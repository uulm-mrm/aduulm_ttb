
#include "gtest/gtest.h"

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/Misc/LockedData.h"

TEST(synchronization, test)
{
  ttb::Synchronized<std::vector<std::vector<double>>> data;
  std::vector<hpx::jthread> threads;
  for (std::size_t i = 0; i < 100; ++i)
  {
    threads.emplace_back([&] {
      hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
      data.wlock().data.emplace_back(1000);
    });
  }
  threads.clear();
  for (std::size_t i = 0; i < 100; ++i)
  {
    threads.emplace_back([i, &data] {
      hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
      for (double& d : data.wlock().data.at(i))
      {
        d++;
      }
    });
  }
  for (std::size_t i = 0; i < 100; ++i)
  {
    threads.emplace_back([i, &data] {
      hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
      for (auto const& ds = data.rlock().data; double const& d : ds.at(i))
      {
        ASSERT_TRUE(true);
      }
    });
  }
  threads.clear();
  std::size_t sum = 0;
  for (auto const& ds : data.rlock().data)
  {
    for (auto const d : ds)
    {
      sum += d;
    }
  }
  ASSERT_EQ(sum, 1000 * 100);
}
