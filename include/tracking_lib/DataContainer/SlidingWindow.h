#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

#include <boost/circular_buffer.hpp>

namespace ttb
{

template <class Data>
struct TimedData
{
  using Data_t = Data;
  Data data;
  Time time;
};

/// A buffer for data with time stamps
/// Data with increasing time stamps can be inserted.
/// Data with older timestamps, based on the window_length, get deleted
/// newer data is at the back of the buffer
template <class Data>
class SlidingWindow
{
public:
  explicit SlidingWindow(Duration window_length) : window_length{ window_length }
  {
  }
  void insert(Time time, Data data)
  {
    assert([&] {
      if (not data_buffer.empty())
      {
        if (time < data_buffer.back().time)
        {
          LOG_FATAL("time of data MUST always increase.");
          return false;
        }
      }
      return true;
    }());
    if (data_buffer.empty())
    {
      data_buffer.push_back({ std::move(data), time });
      return;
    }
    // delete old data
    while (not data_buffer.empty() and time - data_buffer.front().time > window_length)
    {
      data_buffer.pop_front();
    }
    // add to buffer
    if (data_buffer.full())
    {  // no space left, increase capa
      data_buffer.set_capacity(data_buffer.capacity() * 2);
    }
    assert(not data_buffer.full());
    data_buffer.push_back({ std::move(data), time });
    assert([&] {
      for (std::size_t i = 0; i < data_buffer.size() - 1; ++i)
      {
        if (data_buffer.at(i).time > data_buffer.at(i + 1).time)
        {
          return false;
        }
      }
      return true;
    }());
  }

private:
  Duration window_length{ 1s };

public:
  boost::circular_buffer<TimedData<Data>> data_buffer{ 1 };
};

}  // namespace ttb
