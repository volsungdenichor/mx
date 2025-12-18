#include <gtest/gtest.h>
#include <mx/mx.hpp>

// Basic test to verify the test framework is working
TEST(MxTest, BasicTest) {
    EXPECT_TRUE(true);
}

// Test that the mx namespace exists
TEST(MxTest, NamespaceExists) {
    // This test will compile if the namespace is properly defined
    SUCCEED();
}
