
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <string>
#include <tracking_lib/TTBTypes/TTBTypes.h>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

// primary template
template <class T, class...>
struct type
{
  using value_type = T;
};

// specialization for template instantiated types
template <template <class, class...> class T, class F, class... Rest>
struct type<T<F, Rest...>>
{
  using value_type = typename type<F>::value_type;
};

// helper alias
template <class... Ts>
using type_t = typename type<Ts...>::value_type;

template <>
struct type_caster<ttb::ObjectId>
{
  NB_TYPE_CASTER(ttb::ObjectId, const_name("ObjectId"))

  bool from_python(handle src, uint8_t, cleanup_list*) noexcept
  {
    size_t size_p = PyLong_AsSize_t(src.ptr());
    if (size_p == (size_t)-1)
    {
      PyErr_Clear();
      return false;
    }
    value = ttb::ObjectId(size_p);
    return true;
  }

  static handle from_cpp(const ttb::ObjectId& value, rv_policy, cleanup_list*) noexcept
  {
    return PyLong_FromSize_t(value.value_);
  }
};

template <>
struct type_caster<ttb::MeasurementId>
{
  NB_TYPE_CASTER(ttb::MeasurementId, const_name("MeasurementId"))

  bool from_python(handle src, uint8_t, cleanup_list*) noexcept
  {
    size_t size_p = PyLong_AsSize_t(src.ptr());
    if (size_p == (size_t)-1)
    {
      PyErr_Clear();
      return false;
    }
    value = ttb::MeasurementId(size_p);
    return true;
  }

  static handle from_cpp(const ttb::MeasurementId& value, rv_policy, cleanup_list*) noexcept
  {
    return PyLong_FromSize_t(value.value_);
  }
};

template <>
struct type_caster<ttb::StateId>
{
  NB_TYPE_CASTER(ttb::StateId, const_name("StateId"))

  bool from_python(handle src, uint8_t, cleanup_list*) noexcept
  {
    size_t size_p = PyLong_AsSize_t(src.ptr());
    if (size_p == (size_t)-1)
    {
      PyErr_Clear();
      return false;
    }
    value = ttb::StateId(size_p);
    return true;
  }

  static handle from_cpp(const ttb::StateId& value, rv_policy, cleanup_list*) noexcept
  {
    return PyLong_FromSize_t(value.value_);
  }
};

template <>
struct type_caster<ttb::Label>
{
  NB_TYPE_CASTER(ttb::Label, const_name("Label"))

  bool from_python(handle src, uint8_t, cleanup_list*) noexcept
  {
    size_t size_p = PyLong_AsSize_t(src.ptr());
    if (size_p == (size_t)-1)
    {
      PyErr_Clear();
      return false;
    }
    value = ttb::Label(size_p);
    return true;
  }

  static handle from_cpp(const ttb::Label& value, rv_policy, cleanup_list*) noexcept
  {
    return PyLong_FromSize_t(value.value_);
  }
};

template <>
struct type_caster<ttb::MeasModelId>
{
  NB_TYPE_CASTER(ttb::MeasModelId, const_name("MeasModelId"))

  bool from_python(handle src, uint8_t, cleanup_list*) noexcept
  {
    Py_ssize_t size;
    const char* str = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
    if (!str)
    {
      PyErr_Clear();
      return false;
    }
    value = ttb::MeasModelId(std::string(str, (size_t)size));
    return true;
  }

  static handle from_cpp(const ttb::MeasModelId& value, rv_policy, cleanup_list*) noexcept
  {
    return PyUnicode_FromStringAndSize(value.value_.c_str(), value.value_.size());
  }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)