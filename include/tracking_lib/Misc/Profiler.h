#pragma once

#include <aduulm_logger/aduulm_logger.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

namespace ttb::profiling
{

/// Profiler for a general struct
/// It assumes: to_string(struct) -> string
///             to_stringStatistics(std::vector<struct>) -> string
/// It stores all data and the statistics in the given file
template <class Data>
class GeneralDataProfiler
{
public:
  explicit GeneralDataProfiler(std::filesystem::path file) : _file{ std::move(file) }
  {
  }
  void addData(Data&& data)
  {
    std::scoped_lock lock(_datasMutex);
    _datas.emplace_back(std::move(data));
  }
  ~GeneralDataProfiler()
  {
    if (not _datas.empty())
    {
      std::ofstream file(_file.string() + ".stats");
      for (auto const& data : _datas)
      {
        file << to_string(data);
      }
      std::string stats = to_stringStatistics(_datas);
      file << stats;
      LOG_INF(stats);
    }
  }
  std::filesystem::path _file;
  std::mutex _datasMutex;
  std::vector<Data> _datas;
};

}  // namespace ttb::profiling