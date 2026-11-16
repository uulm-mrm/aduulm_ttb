#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include "tracking_lib/Misc/HPXWrapper.h"

namespace ttb::profiling
{

class GeneralDataProfiler
{
public:
  explicit GeneralDataProfiler(std::filesystem::path file);
  void addData(std::string data);
  ~GeneralDataProfiler();
  std::filesystem::path _file;
  hpx::mutex _datasMutex;
  std::vector<std::string> _datas;
};

}  // namespace ttb::profiling
