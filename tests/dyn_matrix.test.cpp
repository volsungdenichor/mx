#include <gmock/gmock.h>

#include <mx/dyn_matrix.hpp>

TEST(dyn_matrix, creation)
{
    mx::dyn_matrix<mx::byte, 3> array{ { 3, 4, 3 } };
    EXPECT_THAT(array.shape(), (mx::dim_t<3>{ mx::dim_base_t{ 3, 12 }, mx::dim_base_t{ 4, 3 }, mx::dim_base_t{ 3, 1 } }));
    EXPECT_THAT(array.volume(), 36);
    EXPECT_THAT(
        array.bounds(),
        (mx::box_shape<mx::size_base_t, 3>{ { mx::interval<mx::size_base_t>{ 0, 3 },
                                              mx::interval<mx::size_base_t>{ 0, 4 },
                                              mx::interval<mx::size_base_t>{ 0, 3 } } }));

    EXPECT_THAT(mx::size(array.bounds()), (mx::size_t<3>{ 3, 4, 3 }));

    {
        auto ref = array.mut_ref();
        ref[{ 0, 0, 0 }] = 255;
        ref[{ 0, 0, 1 }] = 128;
        ref[{ 0, 0, 2 }] = 64;

        ref[{ 0, 2, 0 }] = 10;
        ref[{ 0, 2, 1 }] = 20;
        ref[{ 0, 2, 2 }] = 30;
    }
    {
        auto ref = array.ref();
        EXPECT_THAT((ref[{ 0, 0, 0 }]), 255);
        EXPECT_THAT((ref[{ 0, 0, 1 }]), 128);
        EXPECT_THAT((ref[{ 0, 0, 2 }]), 64);

        EXPECT_THAT((ref[{ 0, 2, 0 }]), 10);
        EXPECT_THAT((ref[{ 0, 2, 1 }]), 20);
        EXPECT_THAT((ref[{ 0, 2, 2 }]), 30);
    }

    {
        const auto slice = array.ref().slice(
            { mx::slice_base_t(0, mx::_), mx::slice_base_t(0, mx::_, 2), mx::slice_base_t{ mx::_, mx::_, -1 } });
        EXPECT_THAT(
            slice.shape(), (mx::dim_t<3>{ mx::dim_base_t{ 3, 12 }, mx::dim_base_t{ 2, 6 }, mx::dim_base_t{ 3, -1 } }));

        EXPECT_THAT((slice[{ 0, 0, 0 }]), 64);
        EXPECT_THAT((slice[{ 0, 0, 1 }]), 128);
        EXPECT_THAT((slice[{ 0, 0, 2 }]), 255);

        EXPECT_THAT((slice[{ 0, 1, 0 }]), 30);
        EXPECT_THAT((slice[{ 0, 1, 1 }]), 20);
        EXPECT_THAT((slice[{ 0, 1, 2 }]), 10);
    }
}
