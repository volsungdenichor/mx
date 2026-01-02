#include <iostream>
#include <mx/dyn_matrix.hpp>
#include <string_view>

auto pixel(mx::location_t<2> loc) -> mx::slice_t<3>
{
    return mx::slice_t<3>{ mx::slice_base_t{ loc[0], loc[0] + 1 },
                           mx::slice_base_t{ loc[1], loc[1] + 1 },
                           mx::slice_base_t{} };
}

void run(const std::vector<std::string_view>& args)
{
    mx::dyn_matrix<mx::byte, 3> image = mx::load_bitmap(std::string{ IMAGES_DIR } + "/lego.bmp");
    std::cout << mx::size(image.bounds()) << std::endl;
    auto ref = image.ref().slice(pixel({ 0, 0 }));
    std::cout << (int)ref[mx::location_t<3>{ 0, 0, 0 }] << " " << (int)ref[mx::location_t<3>{ 0, 0, 1 }] << " "
              << (int)ref[mx::location_t<3>{ 0, 0, 2 }] << std::endl;
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
