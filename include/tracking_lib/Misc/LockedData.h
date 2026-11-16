#pragma once

#include "tracking_lib/Misc/HPXWrapper.h"
#include <tracy/tracy/Tracy.hpp>

namespace ttb
{

template <class Params>
class ConsistentStore
{
public:
  explicit ConsistentStore(Params params)
    : _version{ _global_version.fetch_add(1, std::memory_order_relaxed) }
    , _current{ std::make_shared<Params const>(std::move(params)) }
  {
  }

  ConsistentStore(const ConsistentStore&) = delete;
  ConsistentStore& operator=(const ConsistentStore&) = delete;
  ConsistentStore(ConsistentStore&&) = delete;
  ConsistentStore& operator=(ConsistentStore&&) = delete;

  std::shared_ptr<Params const> get() const
  {
    Cache& cache = _cache;

    uint64_t const v = _version.load(std::memory_order_relaxed);
    if (cache.version == v)
    {
      return cache.ptr;
    }

    auto const fresh = _current.load(std::memory_order_acquire);

    uint64_t const v2 = _version.load(std::memory_order_relaxed);
    if (v == v2)
    {
      cache.ptr = fresh;
      cache.version = v2;
    }

    return fresh;
  }

  void update(Params newParams)
  {
    _current.store(std::make_shared<Params const>(std::move(newParams)), std::memory_order_release);
    _version.store(_global_version.fetch_add(1, std::memory_order_relaxed), std::memory_order_release);
  }

private:
  struct Cache
  {
    std::shared_ptr<Params const> ptr;
    uint64_t version = 0;  // 0 never matches: _global_version starts at 1
  };

  inline static std::atomic<uint64_t> _global_version{ 1 };

  alignas(64) std::atomic<uint64_t> _version;

  alignas(64) std::atomic<std::shared_ptr<Params const>> _current;

  inline static thread_local Cache _cache;
};

template <class T, class M = hpx::shared_mutex>
class LockedWPtr
{
public:
  LockedWPtr(T& t, M* mutex) : data(t), lock(std::unique_lock(*mutex))
  {
  }
  [[nodiscard]] T& operator*() const noexcept
  {
    return data;
  }
  [[nodiscard]] T* operator->() const noexcept
  {
    return &data;
  }
  LockedWPtr(LockedWPtr const& other) = delete;
  LockedWPtr(LockedWPtr&& other) noexcept = delete;
  LockedWPtr& operator=(LockedWPtr const& other) = delete;
  LockedWPtr& operator=(LockedWPtr&& other) = delete;
  T& data;

private:
  std::unique_lock<M> lock;
};

template <class T, class M = hpx::shared_mutex>
class LockedRPtr
{
public:
  LockedRPtr(T const& t, M* mutex) : data(t), lock(std::shared_lock(*mutex))
  {
  }
  LockedRPtr(LockedWPtr<T, M> const& locked_write)
    : data(locked_write.data)  // no need to acquire the lock since it is hold by locked_write
  {
  }
  [[nodiscard]] T const& operator*() const noexcept
  {
    return data;
  }
  [[nodiscard]] T const* operator->() const noexcept
  {
    return &data;
  }
  LockedRPtr(LockedRPtr const& other) = delete;
  LockedRPtr(LockedRPtr&& other) noexcept = delete;
  LockedRPtr& operator=(LockedRPtr const& other) = delete;
  LockedRPtr& operator=(LockedRPtr&& other) = delete;
  T const& data;

private:
  std::shared_lock<M> lock;
};

/// mutex protected data
template <class T>
class Synchronized
{
  mutable hpx::shared_mutex _mutex;
  T _data{};

public:
  Synchronized() = default;
  Synchronized(T data) : _data(std::move(data))
  {
  }
  Synchronized(Synchronized const& other) = delete;
  Synchronized(Synchronized&& other) noexcept = delete;
  Synchronized& operator=(Synchronized const& other) = delete;
  Synchronized& operator=(Synchronized&& other) noexcept = delete;
  ~Synchronized() = default;
  auto wlock()
  {
    return LockedWPtr(_data, &_mutex);
  }

  auto rlock() const
  {
    return LockedRPtr(_data, &_mutex);
  }
};

}  // namespace ttb
