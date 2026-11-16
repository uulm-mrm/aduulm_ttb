#include "tracking_lib/Misc/Profiler.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
namespace ttb::profiling
{

GeneralDataProfiler::GeneralDataProfiler(std::filesystem::path file) : _file{ std::move(file) }
{
}

void GeneralDataProfiler::addData(std::string data)
{
  std::scoped_lock lock(_datasMutex);
  _datas.emplace_back(std::move(data));
}

GeneralDataProfiler::~GeneralDataProfiler()
{
  LOG_FATAL(std::format("Writing Data to {}", _file.c_str()));
  if (not _datas.empty())
  {
    std::ofstream file(_file.string() + ".stats");
    for (std::string const& data : _datas)
    {
      file << data;
    }
  }
}

}  // namespace ttb::profiling