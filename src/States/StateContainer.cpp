#include "tracking_lib/States/StateContainer.h"
#include "tracking_lib/States/State.h"

namespace ttb
{

std::string StateContainer::toString(std::string prefix) const
{
  std::string out = prefix + "StateContainer\n" + prefix + "|\tSource: " + _id.value_ + '\n' + prefix + "|\t" +
                    to_string(_time) + '\n' + _egoMotion.toString(prefix + "|\t");
  for (State const& state : _data)
  {
    out += state.toString(prefix + "|\t");
  }
  return out;
}

}  // namespace ttb