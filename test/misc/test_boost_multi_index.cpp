#include "gtest/gtest.h"

#include <tracking_lib/TTBTypes/TTBTypes.h>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>

TEST(boost, multi_index)
{
  struct Data
  {
    std::size_t unique_id;
    std::size_t label;
    std::vector<std::size_t> values;
  };
  struct Id
  {
  };
  struct Label
  {
  };
  struct Values
  {
  };

  using Container = boost::multi_index::multi_index_container<
      Data,
      boost::multi_index::indexed_by<
          boost::multi_index::hashed_unique<boost::multi_index::tag<Id>,
                                            boost::multi_index::member<Data, std::size_t, &Data::unique_id>>,
          boost::multi_index::hashed_non_unique<boost::multi_index::tag<Label>,
                                                boost::multi_index::member<Data, std::size_t, &Data::label>>,
          boost::multi_index::hashed_unique<
              boost::multi_index::tag<Values>,
              boost::multi_index::member<Data, std::vector<std::size_t>, &Data::values>>>>;
  Data d0{ 0, 66, { 1, 2 } };
  Data d1{ 1, 66, { 3, 5 } };
  Data d2{ 2, 0, { 1, 2 } };
  Container con{ d0, d1 };
  auto [it, inserted] = con.insert(d2);
  EXPECT_EQ(inserted, false);
  EXPECT_EQ(con.size(), 2);
  if (not inserted)
  {
    con.modify(it, [&](auto& elem) { elem.label = d2.label; });
  }
  EXPECT_EQ(con.get<Id>().find(0)->label, 0);
}
