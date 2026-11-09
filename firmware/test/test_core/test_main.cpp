#include <gtest/gtest.h>

// PlatformIO's googletest runner needs an explicit main; the CMake build links this too.
int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
