#include <iostream>
#include <mx/dyn_matrix.hpp>
#include <string_view>

namespace mx
{
std::ostream& operator<<(std::ostream& os, const dyn_matrix_ref<const byte, 1>& item)
{
    os << "[";
    for (std::size_t i = 0; i < item.shape().at(0).size; ++i)
    {
        if (i != 0)
        {
            os << " ";
        }
        os << static_cast<int>(item[{ static_cast<location_base_t>(i) }]);
    }
    os << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const dyn_matrix_ref<byte, 1>& item)
{
    os << "[";
    for (std::size_t i = 0; i < item.shape().at(0).size; ++i)
    {
        if (i != 0)
        {
            os << " ";
        }
        os << static_cast<int>(item[{ static_cast<location_base_t>(i) }]);
    }
    os << "]";
    return os;
}
}  // namespace mx

auto pixel(mx::location_t<2> loc) -> mx::slice_t<3>
{
    return mx::slice_t<3>{ mx::slice_base_t{ loc[0], loc[0] + 1 },
                           mx::slice_base_t{ loc[1], loc[1] + 1 },
                           mx::slice_base_t{} };
}

void set_pixel(mx::dyn_matrix_ref<mx::byte, 3> image, mx::location_t<2> loc, std::array<mx::byte, 3> color)
{
    auto ref = image.slice(pixel(loc)).remove_dim(0).remove_dim(0);
    ref[{ 0 }] = color[0];
    ref[{ 1 }] = color[1];
    ref[{ 2 }] = color[2];
}

void run(const std::vector<std::string_view>& args)
{
    mx::dyn_matrix<mx::byte, 3> image = mx::load_bitmap(std::string{ IMAGES_DIR } + "/lego.bmp");
    set_pixel(image.mut_ref(), { 1, 1 }, { 0, 0, 0 });
    set_pixel(image.mut_ref(), { 1, -2 }, { 255, 0, 255 });
    set_pixel(image.mut_ref(), { -2, -2 }, { 255, 0, 0 });
    set_pixel(image.mut_ref(), { -2, 1 }, { 0, 0, 255 });
    mx::save_bitmap(image, "lego_out.bmp");
    mx::save_bitmap(
        image.ref().slice(mx::slice_t<3>{ mx::slice_base_t{ mx::_, mx::_, -1 }, mx::slice_base_t{}, mx::slice_base_t{} }),
        "lego_out_v_flip.bmp");
    mx::save_bitmap(
        image.ref().slice(mx::slice_t<3>{ mx::slice_base_t{}, mx::slice_base_t{ mx::_, mx::_, -1 }, mx::slice_base_t{} }),
        "lego_out_h_flip.bmp");
}

int main(int argc, char* argv[])
{
    try
    {
        run({ argv + 1, argv + argc });
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
    }

    return 0;
}
