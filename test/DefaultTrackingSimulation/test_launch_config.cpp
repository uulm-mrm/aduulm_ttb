#include "gtest/gtest.h"

#include "tracking_lib/TTBTypes/Params.h"
#include "tracking_lib/TTBManager/TTBManager.h"

#include <filesystem>
#include <figcone/configreader.h>

std::filesystem::path const file{ "test/DefaultTrackingSimulation/tracking_config.yaml" };

TEST(DefaultTrackingSimulation, ReadConfig)
{
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ASSERT_TRUE(true);
}

TEST(DefaultTrackingSimulation, CreateTTBManager)
{
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager manager(std::move(params));
  ASSERT_TRUE(true);
}