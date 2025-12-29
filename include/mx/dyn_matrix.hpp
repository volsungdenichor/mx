#pragma once

#include <cstdint>
#include <fstream>
#include <functional>
#include <mx/mx.hpp>

namespace mx
{

using byte = std::uint8_t;

using location_base_t = std::ptrdiff_t;
using size_base_t = location_base_t;
using stride_base_t = location_base_t;

using flat_offset_t = std::ptrdiff_t;
using volume_t = std::ptrdiff_t;

struct dim_base_t
{
    size_base_t size;
    location_base_t stride;

    dim_base_t() = default;

    dim_base_t(size_base_t size, location_base_t stride) : size(size), stride(stride)
    {
    }

    friend bool operator==(const dim_base_t& lhs, const dim_base_t& rhs)
    {
        return std::tie(lhs.size, lhs.stride) == std::tie(rhs.size, rhs.stride);
    }

    friend bool operator!=(const dim_base_t& lhs, const dim_base_t& rhs)
    {
        return !(lhs == rhs);
    }

    friend std::ostream& operator<<(std::ostream& os, const dim_base_t& item)
    {
        return os << "{"
                  << ":size " << item.size << " "
                  << ":stride " << item.stride << "}";
    }
};

struct slice_base_t
{
    using value_type = std::optional<location_base_t>;

    value_type start;
    value_type stop;
    value_type step;

    slice_base_t() = default;

    slice_base_t(value_type start, value_type stop, value_type step = {}) : start(start), stop(stop), step(step)
    {
    }

    slice_base_t(location_base_t index) : slice_base_t(index, index != -1 ? value_type{ index + 1 } : value_type{})
    {
    }

    friend bool operator==(const slice_base_t& lhs, const slice_base_t& rhs)
    {
        return std::tie(lhs.start, lhs.stop, lhs.step) == std::tie(rhs.start, rhs.stop, rhs.step);
    }

    friend bool operator!=(const slice_base_t& lhs, const slice_base_t& rhs)
    {
        return !(lhs == rhs);
    }

    friend std::ostream& operator<<(std::ostream& os, const slice_base_t& item)
    {
        if (item.start)
        {
            os << *item.start;
        }
        os << ":";
        if (item.stop)
        {
            os << *item.stop;
        }
        os << ":";
        if (item.step)
        {
            os << *item.step;
        }
        return os;
    }
};

template <std::size_t D>
using location_t = vector<location_base_t, D>;

template <std::size_t D>
using size_t = vector<size_base_t, D>;

template <std::size_t D>
using dim_t = vector<dim_base_t, D>;

template <std::size_t D>
using slice_t = vector<slice_base_t, D>;

static constexpr inline auto _ = slice_base_t::value_type{};

namespace detail
{

struct flat_offset_fn
{
    auto operator()(const dim_base_t& item, const location_base_t& loc) const -> flat_offset_t
    {
        return (loc >= 0 ? loc : item.size + loc) * item.stride;
    }

    template <std::size_t D>
    auto operator()(const dim_t<D>& item, const location_t<D>& loc) const -> flat_offset_t
    {
        flat_offset_t result = 0;
        for (std::size_t d = 0; d < D; ++d)
        {
            result += (*this)(item[d], loc[d]);
        }
        return result;
    }
};

static constexpr inline auto flat_offset = flat_offset_fn{};

struct volume_fn
{
    auto operator()(const dim_base_t& item) const -> volume_t
    {
        return item.size;
    }

    template <std::size_t D>
    auto operator()(const dim_t<D>& item) const -> volume_t
    {
        volume_t result = 1;
        for (std::size_t d = 0; d < D; ++d)
        {
            result *= (*this)(item[d]);
        }
        return result;
    }
};

static constexpr inline auto volume = volume_fn{};

struct bounds_fn
{
    template <std::size_t D>
    auto operator()(const dim_t<D>& item) const -> box_shape<size_base_t, D>
    {
        box_shape<size_base_t, D> result;
        for (std::size_t d = 0; d < D; ++d)
        {
            result[d] = { 0, item[d].size };
        }
        return result;
    }
};

static constexpr inline auto bounds = bounds_fn{};

template <std::size_t D>
auto create_shape(const size_t<D>& size) -> dim_t<D>
{
    dim_t<D> result = {};
    stride_base_t stride = 1;
    for (int d = D - 1; d >= 0; --d)
    {
        result[d].size = size[d];
        result[d].stride = stride;
        stride *= size[d];
    }
    return result;
}

inline auto apply_slice(const dim_base_t& dim, const slice_base_t& slice) -> std::tuple<dim_base_t, location_base_t>
{
    const auto step = slice.step.value_or(1);

    if (step == 0)
    {
        throw std::runtime_error{ "step cannot be zero" };
    }
    static const auto ensure_non_negative = [](location_base_t v) { return std::max(location_base_t(0), v); };
    const auto apply_size = [&](location_base_t v) { return v < 0 ? v + dim.size : v; };
    const auto clamp = [&](location_base_t v, location_base_t shift)
    { return std::max(location_base_t(shift), std::min(v, dim.size + shift)); };

    const auto [size, start]  //
        = step > 0            //
              ? std::invoke(
                  [&]() -> std::tuple<size_base_t, location_base_t>
                  {
                      const auto start = clamp(apply_size(slice.start.value_or(0)), 0);
                      const auto stop = clamp(apply_size(slice.stop.value_or(dim.size)), 0);
                      const auto size = ensure_non_negative(((stop - start) + step - 1) / step);
                      return std::tuple{ size, start };
                  })
              : std::invoke(
                  [&]() -> std::tuple<location_base_t, location_base_t>
                  {
                      const auto start = clamp(apply_size(slice.start.value_or(dim.size)), -1);
                      const auto stop = clamp(apply_size(slice.stop.value_or(0)), -1);
                      const auto size = ensure_non_negative(((stop - start) + step) / step);
                      return std::tuple{ size, start };
                  });
    return { dim_base_t{ std::min(size, dim.size), dim.stride * step }, start };
}

template <std::size_t D>
inline auto apply_slice(const dim_t<D>& dim, const slice_t<D>& slice) -> std::tuple<dim_t<D>, location_t<D>>
{
    std::tuple<dim_t<D>, location_t<D>> result;
    for (std::size_t d = 0; d < D; ++d)
    {
        std::tie(std::get<0>(result)[d], std::get<1>(result)[d]) = apply_slice(dim[d], slice[d]);
    }
    return result;
}

}  // namespace detail

template <class T, std::size_t D>
class dyn_matrix_ref
{
public:
    using value_type = std::remove_const_t<T>;
    using shape_type = dim_t<D>;
    using location_type = location_t<D>;
    using slice_type = slice_t<D>;
    using bounds_type = box_shape<size_base_t, D>;
    using reference = T&;
    using pointer = T*;

    dyn_matrix_ref(pointer ptr, shape_type shape) : m_ptr{ ptr }, m_shape{ std::move(shape) }
    {
    }

    const shape_type& shape() const
    {
        return m_shape;
    }

    auto as_const() const -> dyn_matrix_ref<std::add_const_t<T>, D>
    {
        return { m_ptr, m_shape };
    }

    operator dyn_matrix_ref<std::add_const_t<T>, D>() const
    {
        return as_const();
    }

    auto get(const location_type& loc) const -> pointer
    {
        return m_ptr + detail::flat_offset(m_shape, loc);
    }

    auto operator[](const location_type& loc) const -> reference
    {
        return *get(loc);
    }

    auto slice(const slice_type& s) const -> dyn_matrix_ref
    {
        const auto [new_shape, new_loc] = detail::apply_slice(m_shape, s);
        return dyn_matrix_ref{ get(new_loc), new_shape };
    }

    volume_t volume() const
    {
        return detail::volume(m_shape);
    }

    bounds_type bounds() const
    {
        return detail::bounds(m_shape);
    }

    pointer m_ptr;
    shape_type m_shape;
};

template <class T, std::size_t D>
class dyn_matrix
{
public:
    using value_type = T;
    using shape_type = dim_t<D>;
    using location_type = location_t<D>;
    using slice_type = slice_t<D>;
    using bounds_type = box_shape<size_base_t, D>;

    using mut_ref_type = dyn_matrix_ref<T, D>;
    using ref_type = dyn_matrix_ref<const T, D>;
    using reference = typename mut_ref_type::reference;
    using const_reference = typename ref_type::reference;
    using pointer = typename mut_ref_type::pointer;
    using const_pointer = typename ref_type::pointer;

    using data_type = std::vector<T>;

    dyn_matrix(const shape_type& shape, data_type data) : m_shape{ shape }, m_data{ std::move(data) }
    {
    }

    dyn_matrix(const size_t<D>& size) : m_shape{ detail::create_shape(size) }, m_data{}
    {
        m_data.reserve(detail::volume(m_shape));
    }

    const shape_type& shape() const
    {
        return m_shape;
    }

    auto mut_ref() -> mut_ref_type
    {
        return mut_ref_type{ m_data.data(), m_shape };
    }

    auto ref() const -> ref_type
    {
        return ref_type{ m_data.data(), m_shape };
    }

    operator ref_type() const
    {
        return ref();
    }

    operator mut_ref_type()
    {
        return mut_ref();
    }

    volume_t volume() const
    {
        return detail::volume(m_shape);
    }

    bounds_type bounds() const
    {
        return detail::bounds(m_shape);
    }

    shape_type m_shape;
    data_type m_data;
};

namespace detail
{

template <class T, class U = T>
void write(std::ostream& os, const U& value)
{
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <class T>
T read(std::istream& is)
{
    T value;
    is.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

inline void write_n(std::ostream& os, std::size_t count, std::uint8_t value = 0)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        write<std::uint8_t>(os, value);
    }
}

inline auto get_padding(std::size_t width, std::size_t bits_per_pixel) -> std::size_t
{
    return ((bits_per_pixel * width + 31) / 32) * 4 - (width * bits_per_pixel / 8);
}

struct bmp_header
{
    static const inline std::size_t size = 14;

    std::size_t file_size;
    std::size_t data_offset;

    bmp_header() : file_size(0), data_offset(0)
    {
    }

    void save(std::ostream& os) const
    {
        write<std::uint8_t>(os, 'B');
        write<std::uint8_t>(os, 'M');
        write<std::uint32_t>(os, file_size);
        write<std::uint16_t>(os, 0); /* reserved1 */
        write<std::uint16_t>(os, 0); /* reserved2 */
        write<std::uint32_t>(os, data_offset);
    }

    static auto load(std::istream& is) -> bmp_header
    {
        bmp_header result = {};
        read<std::uint8_t>(is);                       /* B */
        read<std::uint8_t>(is);                       /* M */
        result.file_size = read<std::uint32_t>(is);   /**/
        read<std::uint16_t>(is);                      /* reserved1 */
        read<std::uint16_t>(is);                      /* reserved2 */
        result.data_offset = read<std::uint32_t>(is); /* data_offset */
        return result;
    }
};

struct dib_header
{
    static const inline std::size_t size = 40;

    dib_header()
        : width(0)
        , height(0)
        , color_plane_count(0)
        , bits_per_pixel(0)
        , compression(0)
        , data_size(0)
        , horizontal_pixel_per_meter(0)
        , vertical_pixel_per_meter(0)
        , color_count(0)
        , important_color_count(0)
    {
    }

    std::size_t width;
    std::size_t height;
    std::size_t color_plane_count;
    std::size_t bits_per_pixel;
    std::size_t compression;
    std::size_t data_size;
    std::size_t horizontal_pixel_per_meter;
    std::size_t vertical_pixel_per_meter;
    std::size_t color_count;
    std::size_t important_color_count;

    void save(std::ostream& os) const
    {
        write<std::uint32_t>(os, size);
        write<std::uint32_t>(os, width);
        write<std::uint32_t>(os, height);
        write<std::uint16_t>(os, color_plane_count);
        write<std::uint16_t>(os, bits_per_pixel);
        write<std::uint32_t>(os, compression);
        write<std::uint32_t>(os, data_size);
        write<std::uint32_t>(os, horizontal_pixel_per_meter);
        write<std::uint32_t>(os, vertical_pixel_per_meter);
        write<std::uint32_t>(os, color_count);
        write<std::uint32_t>(os, important_color_count);
    }

    static auto load(std::istream& is) -> dib_header
    {
        dib_header result = {};
        read<std::uint32_t>(is); /* size */
        result.width = read<std::uint32_t>(is);
        result.height = read<std::uint32_t>(is);
        result.color_plane_count = read<std::uint16_t>(is);
        result.bits_per_pixel = read<std::uint16_t>(is);
        result.compression = read<std::uint32_t>(is);
        result.data_size = read<std::uint32_t>(is);
        result.horizontal_pixel_per_meter = read<std::uint32_t>(is);
        result.vertical_pixel_per_meter = read<std::uint32_t>(is);
        result.color_count = read<std::uint32_t>(is);
        result.important_color_count = read<std::uint32_t>(is);
        return result;
    }
};

inline void save_header(
    std::ostream& os,  //
    std::size_t width,
    std::size_t height,
    std::size_t padding,
    std::size_t bits_per_pixel,
    std::size_t palette_size)
{
    const std::size_t data_size = (width + padding) * height * (bits_per_pixel / 8);
    const std::size_t data_offset = bmp_header::size + dib_header::size + palette_size;
    const std::size_t file_size = data_offset + data_size;

    bmp_header bmp_hdr = {};
    bmp_hdr.file_size = file_size;
    bmp_hdr.data_offset = data_offset;

    dib_header dib_hdr = {};
    dib_hdr.width = width;
    dib_hdr.height = height;
    dib_hdr.color_plane_count = 1;
    dib_hdr.bits_per_pixel = bits_per_pixel;
    dib_hdr.compression = 0;
    dib_hdr.data_size = data_size;

    bmp_hdr.save(os);
    dib_hdr.save(os);
}

struct load_bitmap_fn
{
    auto operator()(std::istream& is) const -> dyn_matrix<byte, 3>
    {
        if (!is)
        {
            throw std::runtime_error{ "load_bitmap: invalid stream" };
        }

        const bmp_header bmp_hdr = bmp_header::load(is);
        const dib_header dib_hdr = dib_header::load(is);

        switch (dib_hdr.bits_per_pixel)
        {
            case 8: return load_bitmap_8(is, dib_hdr);
            case 24: return load_bitmap_24(is, dib_hdr);
            default: throw std::runtime_error{ "load_bitmap: format not supported" };
        }
    }

    auto operator()(const std::string& path) const -> dyn_matrix<byte, 3>
    {
        std::ifstream fs(path.c_str(), std::ifstream::binary);
        if (!fs)
        {
            throw std::runtime_error{ std::string("load_bitmap: can not load file '" + path + "'") };
        }
        return (*this)(fs);
    }

    static auto prepare_array(const dib_header& header) -> dyn_matrix<byte, 3>
    {
        return dyn_matrix<byte, 3>{ size_t<3>{
            static_cast<size_base_t>(header.height), static_cast<size_base_t>(header.width), 3 } };
    }

    static auto load_bitmap_8(std::istream& is, const dib_header& header) -> dyn_matrix<byte, 3>
    {
        const auto padding = get_padding(header.width, header.bits_per_pixel);

        dyn_matrix<byte, 3> result = prepare_array(header);
        auto ref = result.mut_ref();

        using rgb_t = std::array<byte, 3>;
        using palette_t = std::array<rgb_t, 256>;
        palette_t palette = {};

        for (std::size_t i = 0; i < 256; ++i)
        {
            const byte b = read<byte>(is);
            const byte g = read<byte>(is);
            const byte r = read<byte>(is);
            is.ignore(1);

            palette[i] = rgb_t{ r, g, b };
        }

        const size_base_t h = ref.shape().at(0).size;
        const size_base_t w = ref.shape().at(1).size;

        for (location_base_t y = h - 1; y >= 0; --y)
        {
            for (location_base_t x = 0; x < w; ++x)
            {
                const rgb_t rgb = palette.at(read<byte>(is));
                for (location_base_t z = 0; z < 3; ++z)
                {
                    ref[location_t<3>{ y, x, z }] = rgb[z];
                }
            }

            is.ignore(padding);
        }

        return result;
    }

    static auto load_bitmap_24(std::istream& is, const dib_header& header) -> dyn_matrix<byte, 3>
    {
        const auto padding = get_padding(header.width, header.bits_per_pixel);

        dyn_matrix<byte, 3> result = prepare_array(header);
        auto ref = result.mut_ref();

        const size_base_t h = ref.shape().at(0).size;
        const size_base_t w = ref.shape().at(1).size;

        for (location_base_t y = h - 1; y >= 0; --y)
        {
            for (location_base_t x = 0; x < w; ++x)
            {
                for (location_base_t z = 2; z >= 0; --z)
                {
                    const byte value = read<byte>(is);
                    ref[location_t<3>{ y, x, z }] = value;
                }
            }

            is.ignore(padding);
        }
        return result;
    }
};

struct save_bitmap_fn
{
    void operator()(dyn_matrix_ref<const byte, 2> image, std::ostream& os) const
    {
        static const std::size_t bits_per_pixel = 8;
        const std::size_t padding = get_padding(image.shape().at(1).size, bits_per_pixel);

        const size_base_t h = image.shape().at(0).size;
        const size_base_t w = image.shape().at(1).size;

        save_header(os, w, h, padding, bits_per_pixel, 256 * 4);

        for (std::size_t i = 0; i < 256; ++i)
        {
            for (std::size_t j = 0; j < 3; ++j)
            {
                write<byte>(os, i);
            }
            write<byte>(os, 0);
        }

        for (location_base_t y = h - 1; y >= 0; --y)
        {
            for (location_base_t x = 0; x < w; ++x)
            {
                const byte value = image[location_t<2>{ y, x }];
                write<byte>(os, value);
            }

            write_n(os, padding);
        }
    }

    void operator()(dyn_matrix_ref<const byte, 3> image, std::ostream& os) const
    {
        static const std::size_t bits_per_pixel = 24;
        const std::size_t padding = get_padding(image.shape().at(1).size, bits_per_pixel);

        const size_base_t h = image.shape().at(0).size;
        const size_base_t w = image.shape().at(1).size;

        save_header(os, w, h, padding, bits_per_pixel, 0);

        for (location_base_t y = h - 1; y >= 0; --y)
        {
            for (location_base_t x = 0; x < w; ++x)
            {
                for (location_base_t z = 2; z >= 0; --z)
                {
                    write<byte>(os, image[location_t<3>{ y, x, z }]);
                }
            }

            write_n(os, padding);
        }
    }

    void operator()(dyn_matrix_ref<const byte, 2> image, const std::string& path) const
    {
        std::ofstream fs(path.c_str(), std::ofstream::binary);
        (*this)(image, fs);
    }

    void operator()(dyn_matrix_ref<const byte, 3> image, const std::string& path) const
    {
        std::ofstream fs(path.c_str(), std::ofstream::binary);
        (*this)(image, fs);
    }
};

}  // namespace detail

static constexpr inline auto load_bitmap = detail::load_bitmap_fn{};
static constexpr inline auto save_bitmap = detail::save_bitmap_fn{};

}  // namespace mx
