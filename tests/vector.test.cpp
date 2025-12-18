#include <gmock/gmock.h>

#include <mx/mx.hpp>

template <class M>
constexpr auto ElementAt(std::size_t index, M matcher)
{
    return testing::ResultOf(
        "element [" + std::to_string(index) + "]",
        [index](const auto& container) { return container[index]; },
        std::move(matcher));
}

TEST(vector, vector_elements_access)
{
    EXPECT_THAT(
        (mx::vector_2d<int>{ 1, 2 }),
        testing::AllOf(
            testing::Eq((mx::vector_2d<int>{ 1, 2 })),
            testing::ElementsAre(1, 2),
            ElementAt(0, testing::Eq(1)),
            ElementAt(1, testing::Eq(2))));
}

TEST(vector, vector_plus)
{
    EXPECT_THAT((+mx::vector_2d<int>{ 1, 2 }), testing::Eq((mx::vector_2d<int>{ 1, 2 })));
    EXPECT_THAT((-mx::vector_2d<int>{ 1, 2 }), testing::Eq((mx::vector_2d<int>{ -1, -2 })));
}

TEST(vector, vector_scalar_multiplication)
{
    EXPECT_THAT((mx::vector_2d<int>{ 2, 3 } * 2.5), testing::Eq((mx::vector_2d<double>{ 5.0, 7.5 })));
    EXPECT_THAT((2.5 * mx::vector_2d<int>{ 2, 3 }), testing::Eq((mx::vector_2d<double>{ 5.0, 7.5 })));
}

TEST(vector, vector_scalar_division)
{
    EXPECT_THAT((mx::vector_2d<int>{ 2, 3 } / 2.0), testing::Eq((mx::vector_2d<double>{ 1.0, 1.5 })));
}

TEST(vector, vector_vector_addition)
{
    EXPECT_THAT(
        (mx::vector_2d<int>{ 1, 2 } + mx::vector_2d<double>{ 3.5, 4.5 }), testing::Eq((mx::vector_2d<double>{ 4.5, 6.5 })));
}

TEST(vector, vector_vector_subtraction)
{
    EXPECT_THAT(
        (mx::vector_2d<int>{ 1, 2 } - mx::vector_2d<double>{ 3.5, 4.5 }),
        testing::Eq((mx::vector_2d<double>{ -2.5, -2.5 })));
}
