#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <set>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mx
{

namespace math
{

namespace detail
{

struct sign_fn
{
    template <class T>
    constexpr auto operator()(T v) const -> int
    {
        constexpr T zero = T{};
        if (v > zero)
        {
            return +1;
        }
        else if (v < zero)
        {
            return -1;
        }
        return 0;
    }
};

struct sqr_fn
{
    template <class T, class Res = std::invoke_result_t<std::multiplies<>, T, T>>
    constexpr auto operator()(T v) const -> Res
    {
        return v * v;
    }
};

struct sqrt_fn
{
    template <class T, class Res = decltype(std::sqrt(std::declval<T>()))>
    constexpr auto operator()(T v) const -> Res
    {
        return std::sqrt(v);
    }
};

struct abs_fn
{
    template <class T>
    constexpr auto operator()(T x) const -> T
    {
        return std::abs(x);
    }
};

struct floor_fn
{
    template <class T, class Res = decltype(std::floor(std::declval<T>()))>
    constexpr auto operator()(T x) const -> Res
    {
        return std::floor(x);
    }
};

struct ceil_fn
{
    template <class T, class Res = decltype(std::ceil(std::declval<T>()))>
    constexpr auto operator()(T x) const -> Res
    {
        return std::ceil(x);
    }
};

struct sin_fn
{
    template <class T, class Res = decltype(std::sin(std::declval<T>()))>
    constexpr auto operator()(T x) const -> Res
    {
        return std::sin(x);
    }
};

struct cos_fn
{
    template <class T, class Res = decltype(std::cos(std::declval<T>()))>
    constexpr auto operator()(T x) const -> Res
    {
        return std::cos(x);
    }
};

struct atan2_fn
{
    template <class T, class Res = decltype(std::atan2(std::declval<T>(), std::declval<T>()))>
    constexpr auto operator()(T y, T x) const -> Res
    {
        return std::atan2(y, x);
    }
};

struct asin_fn
{
    template <class T, class Res = decltype(std::asin(std::declval<T>()))>
    constexpr auto operator()(T x) const -> Res
    {
        return std::asin(x);
    }
};

struct acos_fn
{
    template <class T, class Res = decltype(std::acos(std::declval<T>()))>
    constexpr auto operator()(T x) const -> Res
    {
        return std::acos(x);
    }
};

struct binomial_fn
{
    struct cache
    {
        using key_type = std::pair<int, int>;
        using cache_type = std::map<key_type, int>;
        mutable cache_type m_cache;

        static cache& instance()
        {
            static cache inst;
            return inst;
        }

        int operator()(int n, int k) const
        {
            const auto key = key_type{ n, k };

            if (auto it = m_cache.find(key); it != m_cache.end())
            {
                return it->second;
            }

            const int value = k == 0 || k == n ? 1 : (*this)(n - 1, k - 1) + (*this)(n - 1, k);

            m_cache.emplace(key, value);

            return value;
        }
    };

    int operator()(int n, int k) const
    {
        return cache::instance()(n, k);
    }
};

}  // namespace detail

static constexpr inline auto sqr = detail::sqr_fn{};
static constexpr inline auto sqrt = detail::sqrt_fn{};
static constexpr inline auto abs = detail::abs_fn{};
static constexpr inline auto floor = detail::floor_fn{};
static constexpr inline auto ceil = detail::ceil_fn{};
static constexpr inline auto sin = detail::sin_fn{};
static constexpr inline auto cos = detail::cos_fn{};
static constexpr inline auto atans = detail::atan2_fn{};
static constexpr inline auto asin = detail::asin_fn{};
static constexpr inline auto acos = detail::acos_fn{};
static constexpr inline auto sign = detail::sign_fn{};
static const inline auto binomial = detail::binomial_fn{};

}  // namespace math

namespace detail
{

struct delimited_fn
{
    template <class Iter>
    struct impl_t
    {
        Iter m_begin;
        Iter m_end;
        std::string_view m_delimiter;

        constexpr impl_t(Iter begin, Iter end, std::string_view delimiter)
            : m_begin(begin)
            , m_end(end)
            , m_delimiter(delimiter)
        {
        }

        friend std::ostream& operator<<(std::ostream& os, const impl_t& item)
        {
            for (Iter it = item.m_begin; it != item.m_end; ++it)
            {
                if (it != item.m_begin)
                {
                    os << item.m_delimiter;
                }
                os << *it;
            }
            return os;
        }
    };

    template <class Iter>
    constexpr auto operator()(Iter begin, Iter end, std::string_view delimiter) const -> impl_t<Iter>
    {
        return impl_t<Iter>{ begin, end, delimiter };
    }

    template <class Range>
    constexpr auto operator()(Range&& range, std::string_view delimiter) const
    {
        return (*this)(std::begin(range), std::end(range), delimiter);
    }
};
}  // namespace detail

constexpr inline auto delimited = detail::delimited_fn{};

template <class Range, class Out, class Func>
auto transform(Range&& range, Out out, Func&& func) -> Out
{
    return std::transform(std::begin(range), std::end(range), std::move(out), std::forward<Func>(func));
}

template <class Range1, class Range2, class Out, class Func>
auto transform(Range1&& range1, Range2&& range2, Out out, Func&& func) -> Out
{
    return std::transform(
        std::begin(range1), std::end(range1), std::begin(range2), std::move(out), std::forward<Func>(func));
}

template <class Iter>
struct iterator_range
{
    using iterator = Iter;
    using reference = typename std::iterator_traits<iterator>::reference;
    using value_type = typename std::iterator_traits<iterator>::value_type;
    using iterator_category = typename std::iterator_traits<iterator>::iterator_category;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;

    iterator m_begin;
    iterator m_end;

    constexpr iterator_range(iterator begin, iterator end) : m_begin(begin), m_end(end)
    {
    }

    template <
        class It = iterator,
        class = std::enable_if_t<
            std::is_base_of_v<std::random_access_iterator_tag, typename std::iterator_traits<It>::iterator_category>>>
    constexpr iterator_range(iterator begin, difference_type n) : m_begin(begin)
                                                                , m_end(begin + n)
    {
    }

    constexpr iterator begin() const
    {
        return m_begin;
    }

    constexpr iterator end() const
    {
        return m_end;
    }

    template <
        class It = iterator,
        class = std::enable_if_t<
            std::is_base_of_v<std::random_access_iterator_tag, typename std::iterator_traits<It>::iterator_category>>>
    reference operator[](difference_type n) const
    {
        return *(begin() + n);
    }
};

template <class T, class Self>
struct strided_iterator_base
{
    using self_type = Self;
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    T* m_ptr;

    constexpr self_type& self()
    {
        return *static_cast<self_type*>(this);
    }

    constexpr const self_type& self() const
    {
        return *static_cast<const self_type*>(this);
    }

    constexpr reference operator*() const
    {
        return *m_ptr;
    }

    constexpr pointer operator->() const
    {
        return m_ptr;
    }

    constexpr reference operator[](difference_type n) const
    {
        return *(self() + n);
    }

    constexpr self_type& operator++()
    {
        self().increment();
        return self();
    }

    constexpr self_type operator++(int)
    {
        self_type temp = self();
        ++(*this);
        return temp;
    }

    constexpr self_type& operator--()
    {
        self().decrement();
        return self();
    }

    constexpr self_type operator--(int)
    {
        self_type temp = self();
        --(*this);
        return temp;
    }

    constexpr self_type& operator+=(difference_type n)
    {
        self().advance(n);
        return self();
    }

    constexpr self_type operator+(difference_type n) const
    {
        self_type temp = self();
        temp += n;
        return temp;
    }

    constexpr self_type& operator-=(difference_type n)
    {
        return self() += -n;
    }

    constexpr self_type operator-(difference_type n) const
    {
        return self() + (-n);
    }

    friend constexpr bool operator==(const self_type& lhs, const self_type& rhs)
    {
        return lhs.equal_to(rhs);
    }

    friend constexpr bool operator!=(const self_type& lhs, const self_type& rhs)
    {
        return !(lhs == rhs);
    }

    friend constexpr bool operator<(const self_type& lhs, const self_type& rhs)
    {
        return lhs.less_than(rhs);
    }

    friend constexpr bool operator>(const self_type& lhs, const self_type& rhs)
    {
        return rhs < lhs;
    }

    friend constexpr bool operator<=(const self_type& lhs, const self_type& rhs)
    {
        return !(lhs > rhs);
    }

    friend constexpr bool operator>=(const self_type& lhs, const self_type& rhs)
    {
        return !(lhs < rhs);
    }

    friend constexpr std::ptrdiff_t operator-(const self_type& lhs, const self_type& rhs)
    {
        return rhs.distance_to(lhs);
    }
};

template <class T, std::ptrdiff_t N>
struct strided_iterator : strided_iterator_base<T, strided_iterator<T, N>>
{
    static const inline std::ptrdiff_t stride = N;

    constexpr explicit strided_iterator(T* ptr) : strided_iterator_base<T, strided_iterator>{ ptr }
    {
    }

    constexpr strided_iterator() : strided_iterator(nullptr)
    {
    }

    constexpr void increment()
    {
        this->m_ptr += stride;
    }

    constexpr void decrement()
    {
        this->m_ptr -= stride;
    }

    constexpr void advance(std::ptrdiff_t n)
    {
        this->m_ptr += n * stride;
    }

    constexpr bool equal_to(const strided_iterator& rhs) const
    {
        return this->m_ptr == rhs.m_ptr;
    }

    constexpr bool less_than(const strided_iterator& rhs) const
    {
        return stride > 0 ? this->m_ptr < rhs.m_ptr : this->m_ptr > rhs.m_ptr;
    }

    constexpr std::ptrdiff_t distance_to(const strided_iterator& rhs) const
    {
        return (rhs.m_ptr - this->m_ptr) / stride;
    }
};

template <class T>
struct strided_iterator<T, 0> : strided_iterator_base<T, strided_iterator<T, 0>>
{
    std::ptrdiff_t m_stride;

    constexpr strided_iterator(T* ptr, std::ptrdiff_t stride)
        : strided_iterator_base<T, strided_iterator>{ ptr }
        , m_stride{ stride }
    {
    }

    constexpr strided_iterator() : strided_iterator(nullptr, 1)
    {
    }

    constexpr void increment()
    {
        this->m_ptr += m_stride;
    }

    constexpr void decrement()
    {
        this->m_ptr -= m_stride;
    }

    constexpr void advance(std::ptrdiff_t n)
    {
        this->m_ptr += n * m_stride;
    }

    constexpr bool equal_to(const strided_iterator& rhs) const
    {
        return this->m_ptr == rhs.m_ptr;
    }

    constexpr bool less_than(const strided_iterator& rhs) const
    {
        assert(m_stride == rhs.m_stride);
        return m_stride > 0 ? this->m_ptr < rhs.m_ptr : this->m_ptr > rhs.m_ptr;
    }

    constexpr std::ptrdiff_t distance_to(const strided_iterator& rhs) const
    {
        assert(m_stride == rhs.m_stride);
        return (rhs.m_ptr - this->m_ptr) / m_stride;
    }
};

template <class T, std::size_t D>
struct vector;

template <class T, std::size_t R, std::size_t C>
struct matrix_view;

template <class T, std::size_t R, std::size_t C>
struct matrix;

template <class T, std::size_t D>
using square_matrix = matrix<T, D, D>;

template <class T, std::size_t D>
struct vector : public std::array<T, D>
{
    using base_t = std::array<T, D>;

    using base_t::base_t;

    template <class... Tail>
    constexpr vector(T head, Tail... tail) : base_t{ head, static_cast<T>(tail)... }
    {
        static_assert(sizeof...(tail) + 1 == D, "Invalid number of arguments to vector constructor");
    }

    friend std::ostream& operator<<(std::ostream& os, const vector& item)
    {
        return os << "[" << delimited(item, " ") << "]";
    }
};

template <class T>
vector(T, T) -> vector<T, 2>;

template <class T>
vector(T, T, T) -> vector<T, 3>;

template <class T, std::size_t D>
constexpr auto operator+(const vector<T, D>& item) -> vector<T, D>
{
    return item;
}

template <class T, std::size_t D>
constexpr auto operator-(const vector<T, D>& item) -> vector<T, D>
{
    vector<T, D> result;
    mx::transform(item, std::begin(result), std::negate<>{});
    return result;
}

template <class L, class R, std::size_t D, class Res = std::invoke_result_t<std::plus<>, L, R>>
constexpr auto operator+=(vector<L, D>& lhs, const vector<R, D>& rhs) -> vector<L, D>&
{
    mx::transform(lhs, rhs, std::begin(lhs), std::plus<>{});
    return lhs;
}

template <class L, class R, std::size_t D, class Res = std::invoke_result_t<std::plus<>, L, R>>
constexpr auto operator+(const vector<L, D>& lhs, const vector<R, D>& rhs) -> vector<Res, D>
{
    vector<Res, D> result;
    mx::transform(lhs, rhs, std::begin(result), std::plus<>{});
    return result;
}

template <class L, class R, std::size_t D, class Res = std::invoke_result_t<std::minus<>, L, R>>
constexpr auto operator-=(vector<L, D>& lhs, const vector<R, D>& rhs) -> vector<L, D>&
{
    mx::transform(lhs, rhs, std::begin(lhs), std::minus<>{});
    return lhs;
}

template <class L, class R, std::size_t D, class Res = std::invoke_result_t<std::minus<>, L, R>>
constexpr auto operator-(const vector<L, D>& lhs, const vector<R, D>& rhs) -> vector<Res, D>
{
    vector<Res, D> result;
    mx::transform(lhs, rhs, std::begin(result), std::minus<>{});
    return result;
}

template <class L, class R, std::size_t D, class Res = std::invoke_result_t<std::multiplies<>, L, R>>
constexpr auto operator*=(vector<L, D>& lhs, R rhs) -> vector<L, D>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::multiplies<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class L, class R, std::size_t D, class Res = std::invoke_result_t<std::multiplies<>, L, R>>
constexpr auto operator*(const vector<L, D>& lhs, R rhs) -> vector<Res, D>
{
    vector<Res, D> result;
    mx::transform(lhs, std::begin(result), std::bind(std::multiplies<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class L, class R, std::size_t D, class Res = std::invoke_result_t<std::multiplies<>, L, R>>
constexpr auto operator*(L lhs, const vector<R, D>& rhs) -> vector<Res, D>
{
    return rhs * lhs;
}

template <class L, class R, std::size_t D, class Res = std::invoke_result_t<std::divides<>, L, R>>
constexpr auto operator/=(vector<L, D>& lhs, R rhs) -> vector<L, D>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::divides<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class L, class R, std::size_t D, class Res = std::invoke_result_t<std::divides<>, L, R>>
constexpr auto operator/(const vector<L, D>& lhs, R rhs) -> vector<Res, D>
{
    vector<Res, D> result;
    mx::transform(lhs, std::begin(result), std::bind(std::divides<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class L, class R, std::size_t D, class = std::invoke_result_t<std::equal_to<>, L, R>>
constexpr bool operator==(const vector<L, D>& lhs, const vector<R, D>& rhs)
{
    return std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs));
}

template <class L, class R, std::size_t D, class = std::invoke_result_t<std::equal_to<>, L, R>>
constexpr bool operator!=(const vector<L, D>& lhs, const vector<R, D>& rhs)
{
    return !(lhs == rhs);
}

template <class T>
struct interval : public std::array<T, 2>
{
    using base_t = std::array<T, 2>;
    using base_t::base_t;

    interval(T lo, T up) : base_t{ lo, up }
    {
    }

    interval() : interval(T{}, T{})
    {
    }

    friend std::ostream& operator<<(std::ostream& os, const interval& item)
    {
        return os << "[" << item[0] << " " << item[1] << ")";
    }
};

template <class T, class U>
constexpr bool operator==(const interval<T>& lhs, const interval<U>& rhs)
{
    return std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs));
}

template <class T, class U>
constexpr bool operator!=(const interval<T>& lhs, const interval<U>& rhs)
{
    return !(lhs == rhs);
}

template <class T, class U, class = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator+=(interval<T>& lhs, U rhs) -> interval<T>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::plus<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class T, class U, class Res = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator+(const interval<T>& lhs, U rhs) -> interval<Res>
{
    interval<Res> result;
    mx::transform(lhs, std::begin(result), std::bind(std::plus<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class T, class U, class Res = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator+(T lhs, const interval<U>& rhs) -> interval<Res>
{
    return rhs + lhs;
}

template <class T, class U, class = std::invoke_result_t<std::minus<>, T, U>>
constexpr auto operator-=(interval<T>& lhs, U rhs) -> interval<T>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::minus<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class T, class U, class Res = std::invoke_result_t<std::minus<>, T, U>>
constexpr auto operator-(const interval<T>& lhs, U rhs) -> interval<Res>
{
    interval<Res> result;
    mx::transform(lhs, std::begin(result), std::bind(std::minus<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class T, std::size_t D>
struct box_shape : std::array<interval<T>, D>
{
    using base_t = std::array<interval<T>, D>;

    friend std::ostream& operator<<(std::ostream& os, const box_shape& item)
    {
        os << "(";
        for (std::size_t d = 0; d < D; ++d)
        {
            if (d != 0)
            {
                os << " ";
            }
            os << item[d];
        }
        os << ")";
        return os;
    }
};

template <class T, class U, std::size_t D>
constexpr bool operator==(const box_shape<T, D>& lhs, const box_shape<U, D>& rhs)
{
    for (std::size_t d = 0; d < D; ++d)
    {
        if (lhs[d] != rhs[d])
        {
            return false;
        }
    }
    return true;
}

template <class T, class U, std::size_t D>
constexpr bool operator!=(const box_shape<T, D>& lhs, const box_shape<U, D>& rhs)
{
    return !(lhs == rhs);
}

template <class T, class U, std::size_t D, class Res = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator+=(box_shape<T, D>& lhs, const vector<U, D>& rhs) -> box_shape<T, D>&
{
    for (std::size_t d = 0; d < D; ++d)
    {
        lhs[d] += rhs[d];
    }
    return lhs;
}

template <class T, class U, std::size_t D, class Res = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator+(const box_shape<T, D>& lhs, const vector<U, D>& rhs) -> box_shape<Res, D>
{
    box_shape<Res, D> result;
    for (std::size_t d = 0; d < D; ++d)
    {
        result[d] = lhs[d] + rhs[d];
    }
    return result;
}

template <class T, class U, std::size_t D, class Res = std::invoke_result_t<std::minus<>, T, U>>
constexpr auto operator-=(box_shape<T, D>& lhs, const vector<U, D>& rhs) -> box_shape<T, D>&
{
    for (std::size_t d = 0; d < D; ++d)
    {
        lhs[d] -= rhs[d];
    }
    return lhs;
}

template <class T, class U, std::size_t D, class Res = std::invoke_result_t<std::minus<>, T, U>>
constexpr auto operator-(const box_shape<T, D>& lhs, const vector<U, D>& rhs) -> box_shape<Res, D>
{
    box_shape<Res, D> result;
    for (std::size_t d = 0; d < D; ++d)
    {
        result[d] = lhs[d] - rhs[d];
    }
    return result;
}

template <class T, std::size_t R, std::size_t C>
struct matrix_view
{
    using size_type = std::size_t;
    using volume_type = std::size_t;

    using extents_type = vector<size_type, 2>;
    using location_type = vector<size_type, 2>;

    using pointer = T*;
    using reference = T&;
    using row_type = iterator_range<pointer>;
    using column_type = iterator_range<strided_iterator<T, C>>;
    using data_type = iterator_range<pointer>;

    pointer m_data;

    constexpr matrix_view(pointer data) : m_data(data)
    {
    }

    constexpr size_type row_count() const
    {
        return R;
    }

    constexpr size_type column_count() const
    {
        return C;
    }

    constexpr extents_type extents() const
    {
        return extents_type{ row_count(), column_count() };
    }

    constexpr volume_type volume() const
    {
        return row_count() * column_count();
    }

    constexpr data_type data() const
    {
        return data_type{ m_data, volume() };
    }

    constexpr reference operator[](size_type n) const
    {
        return data()[n];
    }

    constexpr reference operator[](const location_type& loc) const
    {
        return *(m_data + loc[0] * column_count() + loc[1]);
    }

    constexpr row_type row(size_type n) const
    {
        return row_type{ typename row_type::iterator{ m_data + n * column_count() }, column_count() };
    }

    constexpr column_type column(size_type n) const
    {
        return column_type{ typename column_type::iterator{ m_data + n }, row_count() };
    }
};

template <class T, std::size_t R, std::size_t C>
struct matrix
{
    using view_type = matrix_view<T, R, C>;
    using const_view_type = matrix_view<const T, R, C>;

    using size_type = typename view_type::size_type;
    using volume_type = typename view_type::volume_type;

    using extents_type = typename view_type::extents_type;
    using location_type = typename view_type::location_type;

    using pointer = typename view_type::pointer;
    using reference = typename view_type::reference;
    using row_type = typename view_type::row_type;
    using column_type = typename view_type::column_type;
    using data_type = typename view_type::data_type;

    using const_pointer = typename const_view_type::pointer;
    using const_reference = typename const_view_type::reference;
    using const_row_type = typename const_view_type::row_type;
    using const_column_type = typename const_view_type::column_type;
    using const_data_type = typename const_view_type::data_type;

    std::array<T, R * C> m_data;

    constexpr matrix() : m_data{}
    {
    }

    constexpr matrix(std::initializer_list<T> init) : matrix{}
    {
        std::copy(std::begin(init), std::end(init), std::begin(m_data));
    }

    constexpr view_type view()
    {
        return view_type{ m_data.data() };
    }

    constexpr const_view_type view() const
    {
        return const_view_type{ m_data.data() };
    }

    constexpr size_type row_count() const
    {
        return view().row_count();
    }

    constexpr size_type column_count() const
    {
        return view().column_count();
    }

    constexpr volume_type volume() const
    {
        return view().volume();
    }

    constexpr extents_type extents() const
    {
        return view().extents();
    }

    constexpr data_type data()
    {
        return view().data();
    }

    constexpr const_data_type data() const
    {
        return view().data();
    }

    constexpr reference operator[](size_type n)
    {
        return view()[n];
    }

    constexpr const_reference operator[](size_type n) const
    {
        return view()[n];
    }

    constexpr reference operator[](const location_type& loc)
    {
        return view()[loc];
    }

    constexpr const_reference operator[](const location_type& loc) const
    {
        return view()[loc];
    }

    constexpr row_type row(size_type n)
    {
        return view().row(n);
    }

    constexpr const_row_type row(size_type n) const
    {
        return view().row(n);
    }

    constexpr column_type column(size_type n)
    {
        return view().column(n);
    }

    constexpr const_column_type column(size_type n) const
    {
        return view().column(n);
    }
};

template <class T, std::size_t R, std::size_t C>
std::ostream& operator<<(std::ostream& os, const matrix_view<T, R, C>& item)
{
    os << "[";
    for (std::size_t r = 0; r < item.row_count(); ++r)
    {
        os << "[" << delimited(item.row(r), " ") << "]";
    }
    os << "]";
    return os;
}

template <class T, std::size_t R, std::size_t C>
std::ostream& operator<<(std::ostream& os, const matrix<T, R, C>& item)
{
    return os << item.view();
}

template <class T, std::size_t R, std::size_t C>
constexpr auto operator+(const matrix<T, R, C>& item) -> matrix<T, R, C>
{
    return item;
}

template <class T, std::size_t R, std::size_t C>
constexpr auto operator-(const matrix<T, R, C>& item) -> matrix<T, R, C>
{
    matrix<T, R, C> result{};
    mx::transform(item.data(), std::begin(result.data()), std::negate<>{});
    return result;
}

template <class T, class U, std::size_t R, std::size_t C, class = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator+=(matrix<T, R, C>& lhs, const matrix<U, R, C>& rhs) -> matrix<T, R, C>&
{
    mx::transform(lhs.data(), rhs.data(), std::begin(lhs.data()), std::plus<>{});
    return lhs;
}

template <class T, class U, std::size_t R, std::size_t C, class = std::invoke_result_t<std::minus<>, T, U>>
constexpr auto operator-=(matrix<T, R, C>& lhs, const matrix<U, R, C>& rhs) -> matrix<T, R, C>&
{
    mx::transform(lhs.data(), rhs.data(), std::begin(lhs.data()), std::minus<>{});
    return lhs;
}

template <class T, class U, std::size_t R, std::size_t C, class = std::invoke_result_t<std::multiplies<>, T, U>>
constexpr auto operator*=(matrix<T, R, C>& lhs, U rhs) -> matrix<T, R, C>&
{
    mx::transform(lhs.data(), std::begin(lhs.data()), std::bind(std::multiplies<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class T, class U, std::size_t R, std::size_t C, class = std::invoke_result_t<std::divides<>, T, U>>
constexpr auto operator/=(matrix<T, R, C>& lhs, U rhs) -> matrix<T, R, C>&
{
    mx::transform(lhs.data(), std::begin(lhs.data()), std::bind(std::divides<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class T, class U, std::size_t R, std::size_t C, class Res = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator+(const matrix<T, R, C>& lhs, const matrix<U, R, C>& rhs) -> matrix<Res, R, C>
{
    matrix<Res, R, C> result{};
    mx::transform(lhs.data(), rhs.data(), std::begin(result.data()), std::plus<>{});
    return result;
}

template <class T, class U, std::size_t R, std::size_t C, class Res = std::invoke_result_t<std::minus<>, T, U>>
constexpr auto operator-(const matrix<T, R, C>& lhs, const matrix<U, R, C>& rhs) -> matrix<Res, R, C>
{
    matrix<Res, R, C> result{};
    mx::transform(lhs.data(), std::begin(rhs.data()), std::begin(result.data()), std::minus<>{});
    return result;
}

template <class T, class U, std::size_t R, std::size_t C, class Res = std::invoke_result_t<std::multiplies<>, T, U>>
constexpr auto operator*(const matrix<T, R, C>& lhs, U rhs) -> matrix<Res, R, C>
{
    matrix<Res, R, C> result{};
    mx::transform(lhs.data(), std::begin(result.data()), std::bind(std::multiplies<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class T, class U, std::size_t R, std::size_t C, class Res = std::invoke_result_t<std::multiplies<>, T, U>>
constexpr auto operator*(T lhs, const matrix<U, R, C>& rhs) -> matrix<Res, R, C>
{
    return rhs * lhs;
}

template <class T, class U, std::size_t R, std::size_t C, class Res = std::invoke_result_t<std::divides<>, T, U>>
constexpr auto operator/(const matrix<T, R, C>& lhs, U rhs) -> matrix<Res, R, C>
{
    matrix<Res, R, C> result{};
    mx::transform(lhs.data(), std::begin(result.data()), std::bind(std::divides<>{}, std::placeholders::_1, rhs));
    return result;
}

template <
    class T,
    class U,
    std::size_t R,
    std::size_t D,
    std::size_t C,
    class Res = std::invoke_result_t<std::multiplies<>, T, U>>
constexpr auto operator*(const matrix<T, R, D>& lhs, const matrix<U, D, C>& rhs) -> matrix<Res, R, C>
{
    matrix<Res, R, C> result{};

    for (std::size_t r = 0; r < R; ++r)
    {
        const auto lhs_row = lhs.row(r);
        for (std::size_t c = 0; c < C; ++c)
        {
            const auto rhs_col = rhs.column(c);
            result[{ r, c }] = std::inner_product(std::begin(lhs_row), std::end(lhs_row), std::begin(rhs_col), Res{});
        }
    }

    return result;
}

template <class T, class U, std::size_t D, class Res = std::invoke_result_t<std::multiplies<>, T, U>>
constexpr auto operator*(const vector<T, D>& lhs, const square_matrix<U, D + 1>& rhs) -> vector<Res, D>
{
    vector<Res, D> result;

    for (std::size_t d = 0; d < D; ++d)
    {
        result[d] = std::inner_product(std::begin(lhs), std::end(lhs), std::begin(rhs.column(d)), Res{ rhs[{ D, d }] });
    }

    return result;
}

template <class T, class U, std::size_t D, class Res = std::invoke_result_t<std::multiplies<>, T, U>>
constexpr auto operator*(const square_matrix<T, D + 1>& lhs, const vector<U, D>& rhs) -> vector<Res, D>
{
    return rhs * lhs;
}

template <class T, class U, std::size_t D, class = std::invoke_result_t<std::multiplies<>, T, U>>
constexpr auto operator*=(vector<T, D>& lhs, const square_matrix<U, D + 1>& rhs) -> vector<T, D>&
{
    return lhs = lhs * rhs;
}

namespace detail
{

struct minor_fn
{
    template <class T, std::size_t R, std::size_t C>
    auto operator()(const matrix<T, R, C>& item, const vector<std::size_t, 2>& loc) const -> matrix<T, R - 1, C - 1>
    {
        static_assert(R > 1, "minor: invalid row.");
        static_assert(C > 1, "minor: invalid col.");

        if (loc[0] >= R || loc[1] >= C)
        {
            throw std::runtime_error{ "minor: invalid row or column" };
        }

        matrix<T, R - 1, C - 1> result{};

        for (std::size_t r = 0; r < R; ++r)
        {
            for (std::size_t c = 0; c < C; ++c)
            {
                result[{ r, c }] = item[{ r + (r < loc[0] ? 0 : 1), c + (c < loc[1] ? 0 : 1) }];
            }
        }

        return result;
    }
};

static constexpr inline auto minor = minor_fn{};

struct transpose_fn
{
    template <class T, std::size_t R, std::size_t C>
    auto operator()(const matrix<T, R, C>& item) const -> matrix<T, C, R>
    {
        matrix<T, C, R> result{};

        for (std::size_t r = 0; r < R; ++r)
        {
            const auto row = item.row(r);
            for (std::size_t c = 0; c < C; ++c)
            {
                const auto col = result.column(c);
                std::copy(std::begin(row), std::end(row), std::begin(col));
            }
        }

        return result;
    }
};

static constexpr inline auto transpose = transpose_fn{};

struct determinant_fn
{
    template <class T>
    constexpr auto operator()(const square_matrix<T, 1>& item) const -> T
    {
        return item[{ 0, 0 }];
    }

    template <class T>
    constexpr auto operator()(const square_matrix<T, 2>& item) const -> decltype(std::declval<T>() * std::declval<T>())
    {
        return item[{ 0, 0 }] * item[{ 1, 1 }] - item[{ 0, 1 }] * item[{ 1, 0 }];
    }

    template <class T>
    constexpr auto operator()(const square_matrix<T, 3>& item) const
        -> decltype(std::declval<T>() * std::declval<T>() * std::declval<T>())
    {
        // clang-format off
        return
            + item[{ 0, 0 }] * item[{ 1, 1 }] * item[{ 2, 2 }]
            + item[{ 0, 1 }] * item[{ 1, 2 }] * item[{ 2, 0 }]
            + item[{ 0, 2 }] * item[{ 1, 0 }] * item[{ 2, 1 }]
            - item[{ 0, 2 }] * item[{ 1, 1 }] * item[{ 2, 0 }]
            - item[{ 0, 0 }] * item[{ 1, 2 }] * item[{ 2, 1 }]
            - item[{ 0, 1 }] * item[{ 1, 0 }] * item[{ 2, 2 }];
        // clang-format on
    }

    template <class T, std::size_t D>
    constexpr auto operator()(const square_matrix<T, D>& item) const
    {
        auto sum = T{};

        for (std::size_t i = 0; i < D; ++i)
        {
            sum += (i % 2 == 0 ? +1 : -1) * item[{ 0, i }] * (*this)(minor(item, { 0, i }));
        }

        return sum;
    }
};

static constexpr inline auto determinant = determinant_fn{};

struct invert_fn
{
    template <class T, std::size_t D>
    auto operator()(const square_matrix<T, D>& value) const -> std::optional<square_matrix<T, D>>
    {
        const auto det = determinant(value);

        if (!det)
        {
            return {};
        }

        square_matrix<T, D> result{};

        for (std::size_t r = 0; r < D; ++r)
        {
            for (std::size_t c = 0; c < D; ++c)
            {
                result[{ c, r }] = T((r + c) % 2 == 0 ? 1 : -1) * determinant(minor(value, { r, c })) / det;
            }
        }

        return result;
    }
};

static constexpr inline auto invert = invert_fn{};

struct identity_fn
{
    template <size_t D, class T = double>
    static constexpr square_matrix<T, D> create()
    {
        square_matrix<T, D> result;

        for (std::size_t r = 0; r < D; ++r)
        {
            for (std::size_t c = 0; c < D; ++c)
            {
                result[{ r, c }] = r == c ? T(1) : T(0);
            }
        }

        return result;
    }

    template <class T, std::size_t D>
    constexpr operator square_matrix<T, D>() const
    {
        return create<D, T>();
    }
};

static constexpr inline auto identity = identity_fn{};

struct scale_fn
{
    template <class T>
    square_matrix<T, 3> operator()(const vector<T, 2>& scale) const
    {
        square_matrix<T, 3> result = identity;

        result[{ 0, 0 }] = scale[0];
        result[{ 1, 1 }] = scale[1];

        return result;
    }

    template <class T>
    square_matrix<T, 4> operator()(const vector<T, 3>& scale) const
    {
        square_matrix<T, 4> result = identity;

        result[{ 0, 0 }] = scale[0];
        result[{ 1, 1 }] = scale[1];
        result[{ 2, 2 }] = scale[2];

        return result;
    }
};

struct rotation_fn
{
    template <class T>
    square_matrix<T, 3> operator()(T angle) const
    {
        square_matrix<T, 3> result = identity;

        const auto c = math::cos(angle);
        const auto s = math::sin(angle);

        result[{ 0, 0 }] = c;
        result[{ 0, 1 }] = s;
        result[{ 1, 0 }] = -s;
        result[{ 1, 1 }] = c;

        return result;
    }
};

struct translation_fn
{
    template <class T>
    square_matrix<T, 3> operator()(const vector<T, 2>& offset) const
    {
        square_matrix<T, 3> result = identity;

        result[{ 2, 0 }] = offset[0];
        result[{ 2, 1 }] = offset[1];

        return result;
    }

    template <class T>
    square_matrix<T, 4> operator()(const vector<T, 3>& offset) const
    {
        square_matrix<T, 4> result = identity;

        result[{ 3, 0 }] = offset[0];
        result[{ 3, 1 }] = offset[1];
        result[{ 3, 2 }] = offset[2];

        return result;
    }
};

}  // namespace detail

using detail::determinant;
using detail::invert;
using detail::minor;
using detail::transpose;

static constexpr inline auto scale = detail::scale_fn{};
static constexpr inline auto translation = detail::translation_fn{};
static constexpr inline auto rotation = detail::rotation_fn{};

namespace detail
{

struct ray_tag
{
};
struct line_tag
{
};
struct segment_tag
{
};

}  // namespace detail

template <class Tag, class T, std::size_t D>
struct linear_shape : std::array<vector<T, D>, 2>
{
    using base_t = std::array<vector<T, D>, 2>;

    using base_t::base_t;

    linear_shape(vector<T, D> p0, vector<T, D> p1) : base_t{ { p0, p1 } }
    {
    }
};

template <class T, std::size_t D>
using line = linear_shape<detail::line_tag, T, D>;

template <class T, std::size_t D>
using ray = linear_shape<detail::ray_tag, T, D>;

template <class T, std::size_t D>
using segment = linear_shape<detail::segment_tag, T, D>;

template <class T, std::size_t D>
std::ostream& operator<<(std::ostream& os, const line<T, D>& item)
{
    return os << "(line " << item[0] << " (dir " << (item[1] - item[0]) << "))";
}

template <class T, std::size_t D>
std::ostream& operator<<(std::ostream& os, const ray<T, D>& item)
{
    return os << "(ray " << item[0] << " (dir " << (item[1] - item[0]) << "))";
}

template <class T, std::size_t D>
std::ostream& operator<<(std::ostream& os, const segment<T, D>& item)
{
    return os << "(segment " << item[0] << " " << item[1] << ")";
}

template <class Tag, class T, class U, std::size_t D>
constexpr auto operator+=(linear_shape<Tag, T, D>& lhs, const vector<U, D>& rhs) -> linear_shape<Tag, T, D>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::plus<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class Tag, class T, class U, std::size_t D, class Res = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator+(const linear_shape<Tag, T, D>& lhs, const vector<U, D>& rhs) -> linear_shape<Tag, Res, D>
{
    linear_shape<Tag, Res, D> result;
    mx::transform(lhs, std::begin(result), std::bind(std::plus<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class Tag, class T, class U, std::size_t D>
constexpr auto operator-=(linear_shape<Tag, T, D>& lhs, const vector<U, D>& rhs) -> linear_shape<Tag, T, D>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::minus<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class Tag, class T, class U, std::size_t D, class Res = std::invoke_result_t<std::minus<>, T, U>>
constexpr auto operator-(const linear_shape<Tag, T, D>& lhs, const vector<U, D>& rhs) -> linear_shape<Tag, Res, D>
{
    linear_shape<Tag, Res, D> result;
    mx::transform(lhs, std::begin(result), std::bind(std::minus<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class Tag, class T, class U, std::size_t D>
constexpr auto operator*=(linear_shape<Tag, T, D>& lhs, const square_matrix<U, D + 1>& rhs) -> linear_shape<Tag, T, D>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::multiplies<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class Tag, class T, class U, std::size_t D, class Res = std::invoke_result_t<std::multiplies<>, T, U>>
constexpr auto operator*(const linear_shape<Tag, T, D>& lhs, const square_matrix<U, D + 1>& rhs) -> linear_shape<Tag, Res, D>
{
    linear_shape<Tag, Res, D> result;
    mx::transform(lhs, std::begin(result), std::bind(std::multiplies<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class Tag, class T, class U, std::size_t D, class Res = std::invoke_result_t<std::multiplies<>, T, U>>
constexpr auto operator*(const square_matrix<U, D + 1>& lhs, const linear_shape<Tag, T, D>& rhs) -> linear_shape<Tag, Res, D>
{
    return rhs * lhs;
}

template <class T, std::size_t D, std::size_t N>
struct polygonal_shape : std::array<vector<T, D>, N>
{
    using base_t = std::array<vector<T, D>, N>;

    friend std::ostream& operator<<(std::ostream& os, const polygonal_shape& item)
    {
        os << "(";
        for (std::size_t n = 0; n < item.size(); ++n)
        {
            if (n != 0)
            {
                os << " ";
            }
            os << item[n];
        }
        os << ")";
        return os;
    }
};

template <class T, std::size_t D>
using triangle = polygonal_shape<T, D, 3>;

template <class T, std::size_t D>
using quad = polygonal_shape<T, D, 4>;

template <class T, class U, std::size_t D, std::size_t N>
constexpr auto operator+=(polygonal_shape<T, D, N>& lhs, const vector<U, D>& rhs) -> polygonal_shape<T, D, N>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::plus<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class T, class U, std::size_t D, std::size_t N, class Res = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator+(const polygonal_shape<T, D, N>& lhs, const vector<U, D>& rhs) -> polygonal_shape<Res, D, N>
{
    polygonal_shape<Res, D, N> result;
    mx::transform(lhs, std::begin(result), std::bind(std::plus<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class T, class U, std::size_t D, std::size_t N>
constexpr auto operator-=(polygonal_shape<T, D, N>& lhs, const vector<U, D>& rhs) -> polygonal_shape<T, D, N>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::minus<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class T, class U, std::size_t D, std::size_t N, class Res = std::invoke_result_t<std::minus<>, T, U>>
constexpr auto operator-(const polygonal_shape<T, D, N>& lhs, const vector<U, D>& rhs) -> polygonal_shape<Res, D, N>
{
    polygonal_shape<Res, D, N> result;
    mx::transform(lhs, std::begin(result), std::bind(std::minus<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class T, class U, std::size_t D, std::size_t N>
constexpr auto operator*=(polygonal_shape<T, D, N>& lhs, const square_matrix<U, D + 1>& rhs) -> polygonal_shape<T, D, N>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::multiplies<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class T, class U, std::size_t D, std::size_t N, class Res = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator*(const polygonal_shape<T, D, N>& lhs, const square_matrix<U, D + 1>& rhs)
    -> polygonal_shape<Res, D, N>
{
    polygonal_shape<Res, D, N> result;
    mx::transform(lhs, std::begin(result), std::bind(std::multiplies<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class T, class U, std::size_t D, std::size_t N, class Res = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator*(const square_matrix<U, D + 1>& lhs, const polygonal_shape<T, D, N>& rhs)
    -> polygonal_shape<Res, D, N>
{
    return rhs * lhs;
}

namespace detail
{

struct polygon_tag
{
};

struct polyline_tag
{
};

}  // namespace detail

template <class Tag, class T, std::size_t D>
struct vertex_list_shape : public std::vector<vector<T, D>>
{
    using base_t = std::vector<vector<T, D>>;
    using base_t::base_t;
};

template <class T, std::size_t D>
using polygon = vertex_list_shape<detail::polygon_tag, T, D>;

template <class T, std::size_t D>
using polyline = vertex_list_shape<detail::polyline_tag, T, D>;

template <class T, std::size_t D>
std::ostream& operator<<(std::ostream& os, const polygon<T, D>& item)
{
    return os << "(polygon " << delimited(item, " ") << ")";
}

template <class T, std::size_t D>
std::ostream& operator<<(std::ostream& os, const polyline<T, D>& item)
{
    return os << "(polyline " << delimited(item, " ") << ")";
}

template <class Tag, class T, class U, std::size_t D>
constexpr auto operator+=(vertex_list_shape<Tag, T, D>& lhs, const vector<U, D>& rhs) -> vertex_list_shape<Tag, T, D>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::plus<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class Tag, class T, class U, std::size_t D, class Res = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator+(const vertex_list_shape<Tag, T, D>& lhs, const vector<U, D>& rhs) -> vertex_list_shape<Tag, Res, D>
{
    vertex_list_shape<Tag, Res, D> result(lhs.size());
    mx::transform(lhs, std::begin(result), std::bind(std::plus<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class Tag, class T, class U, std::size_t D>
constexpr auto operator-=(vertex_list_shape<Tag, T, D>& lhs, const vector<U, D>& rhs) -> vertex_list_shape<Tag, T, D>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::minus<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class Tag, class T, class U, std::size_t D, class Res = std::invoke_result_t<std::minus<>, T, U>>
constexpr auto operator-(const vertex_list_shape<Tag, T, D>& lhs, const vector<U, D>& rhs) -> vertex_list_shape<Tag, Res, D>
{
    vertex_list_shape<Tag, Res, D> result(lhs.size());
    mx::transform(lhs, std::begin(result), std::bind(std::minus<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class Tag, class T, class U, std::size_t D>
constexpr auto operator*=(vertex_list_shape<Tag, T, D>& lhs, const square_matrix<U, D + 1>& rhs)
    -> vertex_list_shape<Tag, T, D>&
{
    mx::transform(lhs, std::begin(lhs), std::bind(std::multiplies<>{}, std::placeholders::_1, rhs));
    return lhs;
}

template <class Tag, class T, class U, std::size_t D, class Res = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator*(const vertex_list_shape<Tag, T, D>& lhs, const square_matrix<U, D + 1>& rhs)
    -> vertex_list_shape<Tag, Res, D>
{
    vertex_list_shape<Tag, Res, D> result(lhs.size());
    mx::transform(lhs, std::begin(result), std::bind(std::multiplies<>{}, std::placeholders::_1, rhs));
    return result;
}

template <class Tag, class T, class U, std::size_t D, class Res = std::invoke_result_t<std::plus<>, T, U>>
constexpr auto operator*(const square_matrix<U, D + 1>& lhs, const vertex_list_shape<Tag, T, D>& rhs)
    -> vertex_list_shape<Tag, Res, D>
{
    return rhs * lhs;
}

template <class T, std::size_t D>
struct spherical_shape
{
    vector<T, D> center;
    T radius;
};

template <class T>
std::ostream& operator<<(std::ostream& os, const spherical_shape<T, 2>& item)
{
    return os << "(circle " << item.center << " " << item.radius << ")";
}

template <class T>
std::ostream& operator<<(std::ostream& os, const spherical_shape<T, 3>& item)
{
    return os << "(sphere " << item.center << " " << item.radius << ")";
}

template <class T, class U, std::size_t D>
constexpr auto operator+=(spherical_shape<T, D>& lhs, const vector<U, D>& rhs) -> spherical_shape<T, D>&
{
    lhs.center += rhs;
    return lhs;
}

template <class T, class U, std::size_t D>
constexpr auto operator+(spherical_shape<T, D> lhs, const vector<U, D>& rhs) -> spherical_shape<T, D>
{
    lhs.center += rhs;
    return lhs;
}

template <class T, class U, std::size_t D>
constexpr auto operator-=(spherical_shape<T, D>& lhs, const vector<U, D>& rhs) -> spherical_shape<T, D>&
{
    lhs.center -= rhs;
    return lhs;
}

template <class T, class U, std::size_t D>
constexpr auto operator-(spherical_shape<T, D> lhs, const vector<U, D>& rhs) -> spherical_shape<T, D>
{
    lhs.center -= rhs;
    return lhs;
}

namespace detail
{

template <class T>
constexpr bool between(T v, T lo, T up)
{
    return lo <= v && v < up;
}

template <class T>
constexpr bool inclusive_between(T v, T lo, T up)
{
    return lo <= v && v <= up;
}

template <class T, class E>
constexpr auto approx_equal(T value, E epsilon)
{
    return [=](auto v) { return std::abs(v - value) < epsilon; };
}

template <class T, std::size_t D>
constexpr line<T, 2> make_line(const segment<T, D>& s)
{
    return line<T, 2>{ s[0], s[1] };
}

struct dot_fn
{
    template <class T, class U, std::size_t D, class Res = std::invoke_result_t<std::multiplies<>, T, U>>
    constexpr auto operator()(const vector<T, D>& lhs, const vector<U, D>& rhs) const -> Res
    {
        return std::inner_product(std::begin(lhs), std::end(lhs), std::begin(rhs), Res{});
    }
};

static constexpr inline auto dot = dot_fn{};

struct cross_fn
{
    template <class T, class U, class Res = std::invoke_result_t<std::multiplies<>, T, U>>
    constexpr auto operator()(const vector<T, 2>& lhs, const vector<U, 2>& rhs) const -> Res
    {
        return lhs[0] * rhs[1] - lhs[1] * rhs[0];
    }

    template <class T, class U, class Res = std::invoke_result_t<std::multiplies<>, T, U>>
    constexpr auto operator()(const vector<T, 3>& lhs, const vector<U, 3>& rhs) const -> vector<Res, 3>
    {
        return vector<Res, 3>{ { lhs[1] * rhs[2] - lhs[2] * rhs[1],  //
                                 lhs[2] * rhs[0] - lhs[0] * rhs[2],
                                 lhs[0] * rhs[1] - lhs[1] * rhs[0] } };
    }
};

static constexpr inline auto cross = cross_fn{};

struct angle_fn
{
    template <class T>
    constexpr auto operator()(const vector<T, 2>& lhs, const vector<T, 2>& rhs) const
        -> decltype(atan2(cross(lhs, rhs), dot(lhs, rhs)))
    {
        return atan2(cross(lhs, rhs), dot(lhs, rhs));
    }

    template <class T>
    constexpr auto operator()(const vector<T, 3>& lhs, const vector<T, 3>& rhs) const
        -> decltype(acos(dot(lhs, rhs) / (length(lhs) * length(rhs))))
    {
        return acos(dot(lhs, rhs) / (length(lhs) * length(rhs)));
    }
};

static constexpr inline auto angle = angle_fn{};

struct norm_fn
{
    template <class T, std::size_t D, class Res = std::invoke_result_t<std::multiplies<>, T, T>>
    constexpr auto operator()(const vector<T, D>& item) const -> Res
    {
        return dot(item, item);
    }
};

static constexpr inline auto norm = norm_fn{};

struct length_fn
{
    template <class T, std::size_t D>
    constexpr auto operator()(const vector<T, D>& item) const -> decltype(sqrt(norm(item)))
    {
        return sqrt(norm(item));
    }

    template <class T, std::size_t D>
    constexpr auto operator()(const segment<T, D>& item) const
    {
        return (*this)(item[1] - item[0]);
    }
};

static constexpr inline auto length = length_fn{};

struct unit_fn
{
    template <
        class T,
        std::size_t D,
        class Sqr = std::invoke_result_t<std::multiplies<>, T, T>,
        class Sqrt = decltype(sqrt(std::declval<Sqr>())),
        class Res = std::invoke_result_t<std::divides<>, T, Sqrt>>
    constexpr auto operator()(const vector<T, D>& item) const -> vector<Res, D>
    {
        const auto len = length(item);
        return len ? item / len : item;
    }
};

static constexpr inline auto unit = unit_fn{};

struct distance_fn
{
    template <class T, class U, std::size_t D>
    constexpr auto operator()(const vector<T, D>& lhs, const vector<U, D>& rhs) const -> decltype(length(rhs - lhs))
    {
        return length(rhs - lhs);
    }
};

static constexpr inline auto distance = distance_fn{};

template <std::size_t Dim>
struct lower_upper_fn
{
    template <class T>
    constexpr auto operator()(const interval<T>& item) const -> T
    {
        return item[Dim];
    }

    template <class T, std::size_t D>
    constexpr auto operator()(const box_shape<T, D>& item) const -> vector<T, D>
    {
        vector<T, D> result;
        for (std::size_t d = 0; d < D; ++d)
        {
            result[d] = (*this)(item[d]);
        }
        return result;
    }
};

static constexpr inline auto lower = lower_upper_fn<0>{};
static constexpr inline auto upper = lower_upper_fn<1>{};

template <std::size_t Dim>
struct min_max_fn
{
    template <class T>
    constexpr auto operator()(const interval<T>& item) const -> T
    {
        if constexpr (Dim == 0)
        {
            return item[0];
        }
        else
        {
            if constexpr (std::is_integral_v<T>)
            {
                return item[1] - 1;
            }
            else
            {
                return item[1] - std::numeric_limits<T>::epsilon();
            }
        }
    }

    template <class T, std::size_t D>
    constexpr auto operator()(const box_shape<T, D>& item) const -> vector<T, D>
    {
        vector<T, D> result;
        for (std::size_t d = 0; d < D; ++d)
        {
            result[d] = (*this)(item[d]);
        }
        return result;
    }
};

static constexpr inline auto min = min_max_fn<0>{};
static constexpr inline auto max = min_max_fn<1>{};

struct size_fn
{
    template <class T>
    constexpr auto operator()(const interval<T>& item) const -> T
    {
        return upper(item) - lower(item);
    }

    template <class T, std::size_t D>
    constexpr auto operator()(const box_shape<T, D>& item) const -> vector<T, D>
    {
        return upper(item) - lower(item);
    }
};

static constexpr inline auto size = size_fn{};

struct center_fn
{
    template <class T, std::size_t D>
    constexpr auto operator()(const box_shape<T, D>& item) const -> vector<T, D>
    {
        return (lower(item) + upper(item)) / 2;
    }

    template <class T, std::size_t D>
    constexpr auto operator()(const segment<T, D>& item) const -> vector<T, D>
    {
        return (item[0] + item[1]) / 2;
    }

    template <class T, std::size_t D>
    constexpr auto operator()(const spherical_shape<T, D>& item) const -> vector<T, D>
    {
        return item.center;
    }
};

static constexpr inline auto center = center_fn{};

struct orientation_fn
{
    template <class T, class U>
    constexpr auto operator()(const vector<T, 2>& point, const vector<U, 2>& start, const vector<U, 2>& end) const
    {
        return cross(end - start, point - start);
    }

    template <class T, class U, class Tag>
    constexpr auto operator()(const vector<T, 2>& point, const linear_shape<Tag, U, 2>& shape) const
    {
        return (*this)(point, shape[0], shape[1]);
    }
};

static constexpr inline auto orientation = orientation_fn{};

struct contains_fn
{
    template <class T, class U>
    constexpr auto operator()(const interval<T>& item, U value) const -> bool
    {
        return between(value, lower(item), upper(item));
    }

    template <class T>
    constexpr auto operator()(const interval<T>& item, const interval<T>& other) const -> bool
    {
        constexpr T lo = lower(item);
        constexpr T up = upper(item);
        return inclusive_between(lower(other), lo, up) && inclusive_between(upper(other), lo, up);
    }

    template <class T, std::size_t D>
    constexpr auto operator()(const box_shape<T, D>& item, const box_shape<T, D>& other) const -> bool
    {
        for (std::size_t d = 0; d < D; ++d)
        {
            if (!(*this)(item[d], other[d]))
            {
                return false;
            }
        }
        return true;
    }

    template <class T, class U, std::size_t D>
    constexpr auto operator()(const spherical_shape<T, D>& item, const vector<U, D>& other) const -> bool
    {
        return norm(other - center(item)) <= sqr(item.radius);
    }

    template <class T, class U>
    constexpr bool operator()(const triangle<T, 2>& item, const vector<U, 2>& other) const
    {
        constexpr auto same_sign = [](int a, int b) { return (a <= 0 && b <= 0) || (a >= 0 && b >= 0); };

        int result[3];

        for (std::size_t i = 0; i < 3; ++i)
        {
            result[i] = sign(orientation(other, segment<T, 2>{ item[(i + 0) % 3], item[(i + 1) % 3] }));
        }

        return same_sign(result[0], result[1]) && same_sign(result[0], result[2]) && same_sign(result[1], result[2]);
    }
};

static constexpr inline auto contains = contains_fn{};

struct intersects_fn
{
    template <class T>
    constexpr auto operator()(const interval<T>& self, const interval<T>& other) const -> bool
    {
        return inclusive_between(lower(self), lower(other), upper(other))     //
               || inclusive_between(upper(self), lower(other), upper(other))  //
               || inclusive_between(lower(other), lower(self), upper(self))   //
               || inclusive_between(upper(other), lower(self), upper(self));
    }

    template <class T, std::size_t D>
    constexpr auto operator()(const box_shape<T, D>& self, const box_shape<T, D>& other) const -> bool
    {
        for (std::size_t d = 0; d < D; ++d)
        {
            if (!(*this)(self[d], other[d]))
            {
                return false;
            }
        }
        return true;
    }
};

static constexpr inline auto intersects = intersects_fn{};

struct interpolate_fn
{
    template <class R, class T, std::size_t D>
    constexpr auto operator()(R r, const vector<T, D>& lhs, const vector<T, D>& rhs) const -> vector<T, D>
    {
        return (lhs * r) + (rhs * (R(1) - r));
    }

    template <class R, class T, std::size_t D>
    constexpr auto operator()(R r, const segment<T, D>& value) const
    {
        return (*this)(r, value[0], value[1]);
    }

    template <class R, class T>
    constexpr auto operator()(R r, const interval<T>& item) const
    {
        return lower(item) * r + upper(item) * (R(1) - r);
    }
};

static constexpr inline auto interpolate = interpolate_fn{};

namespace detail
{

template <class T, class E>
constexpr auto get_line_intersection_parameter(
    const vector<T, 2>& a0, const vector<T, 2>& a1, const vector<T, 2>& p, E epsilon) -> std::optional<T>
{
    const auto dir = a1 - a0;

    const auto d = p - a0;

    const auto det = cross(dir, d);
    if (approx_equal(E(0), epsilon)(det))
    {
        return {};
    }

    return dot(d, dir) / norm(dir);
}

template <class T, class E>
constexpr auto get_line_intersection_parameters(
    const vector<T, 2>& a0, const vector<T, 2>& a1, const vector<T, 2>& b0, const vector<T, 2>& b1, E epsilon)
    -> std::optional<std::tuple<T, T>>
{
    const auto dir_a = a1 - a0;
    const auto dir_b = b1 - b0;

    const auto det = cross(dir_a, dir_b);
    const auto v = b0 - a0;

    if (approx_equal(E(0), epsilon)(det))
    {
        return {};
    }

    return { { cross(v, dir_b) / det, cross(v, dir_a) / det } };
}

template <class T>
constexpr bool contains_param(line_tag, T)
{
    return true;
}

template <class T>
constexpr bool contains_param(ray_tag, T v)
{
    return v >= T(0);
}

template <class T>
constexpr bool contains_param(segment_tag, T v)
{
    return T(0) <= v && v <= T(1);
}

}  // namespace detail

struct intersection_fn
{
    template <class T, std::size_t D, class Tag1, class Tag2, class E = T>
    constexpr auto operator()(const linear_shape<Tag1, T, D>& lhs, const linear_shape<Tag2, T, D>& rhs, E epsilon = {}) const
        -> std::optional<vector<T, D>>
    {
        const auto par = detail::get_line_intersection_parameters(lhs[0], lhs[1], rhs[0], rhs[1], epsilon);

        if (!par)
        {
            return {};
        }

        const auto [a, b] = *par;

        if (detail::contains_param(Tag1{}, a) && detail::contains_param(Tag2{}, b))
        {
            return interpolate(a, lhs[0], lhs[1]);
        }
        return {};
    }
};

static constexpr inline auto intersection = intersection_fn{};

struct projection_fn
{
    template <class T, std::size_t D>
    constexpr auto operator()(const vector<T, D>& lhs, const vector<T, D>& rhs) const
        -> decltype(rhs * (dot(rhs, lhs) / norm(rhs)))
    {
        return rhs * (dot(rhs, lhs) / norm(rhs));
    }

    template <class T, std::size_t D, class Tag, class E = T>
    constexpr auto operator()(const vector<T, D>& point, const linear_shape<Tag, T, D>& shape, E epsilon = {}) const
        -> std::optional<vector<T, D>>
    {
        const auto p0 = shape[0];
        const auto p1 = shape[1];

        const auto result = p0 + (*this)(point - p0, p1 - p0);

        const auto t = detail::get_line_intersection_parameter(p0, p1, result, epsilon);

        if (t && detail::contains_param(Tag{}, *t))
        {
            return result;
        }

        return {};
    }
};

static constexpr inline auto projection = projection_fn{};

struct rejection_fn
{
    template <class T, std::size_t D>
    constexpr auto operator()(const vector<T, D>& lhs, const vector<T, D>& rhs) const -> decltype(lhs - projection(lhs, rhs))
    {
        return lhs - projection(lhs, rhs);
    }
};

static constexpr inline auto rejection = rejection_fn{};

struct perpendicular_fn
{
    template <class T>
    constexpr auto operator()(const vector<T, 2>& value) const -> vector<T, 2>
    {
        return vector<T, 2>{ -value[1], value[0] };
    }

    template <class Tag, class T>
    constexpr auto operator()(const linear_shape<Tag, T, 2>& value, const vector<T, 2>& origin) const
        -> linear_shape<Tag, T, 2>
    {
        return { origin, origin + (*this)(value[1] - value[0]) };
    }

    template <class Tag, class T>
    constexpr auto operator()(const linear_shape<Tag, T, 2>& value) const -> linear_shape<Tag, T, 2>
    {
        return (*this)(value, value[0]);
    }
};

static constexpr inline auto perpendicular = perpendicular_fn{};

struct altitude_fn
{
    template <typename T>
    constexpr auto operator()(const triangle<T, 2>& value, std::size_t index) const -> segment<T, 2>
    {
        constexpr T epsilon = T(0.1);

        const auto v = value[(index + 0) % 3];

        const auto p = projection(v, line<T, 2>{ value[(index + 1) % 3], value[(index + 2) % 3] }, epsilon);

        return { v, *p };
    }
};

static constexpr inline auto altitude = altitude_fn{};

struct centroid_fn
{
    template <typename T>
    constexpr auto operator()(const triangle<T, 2>& value) const -> vector<T, 2>
    {
        return std::accumulate(std::begin(value), std::end(value), vector<T, 2>{}) / 3;
    }
};

static constexpr inline auto centroid = centroid_fn{};

struct orthocenter_fn
{
    template <typename T>
    constexpr auto operator()(const triangle<T, 2>& value) const -> vector<T, 2>
    {
        constexpr T epsilon = T(0.0001);

        return *intersection(make_line(altitude(value, 0)), make_line(altitude(value, 1)), epsilon);
    }
};

static constexpr inline auto orthocenter = orthocenter_fn{};

struct circumcenter_fn
{
    template <typename T>
    constexpr auto operator()(const triangle<T, 2>& value) const -> vector<T, 2>
    {
        constexpr T epsilon = T(0.0001);

        const auto s0 = segment<T, 2>{ value[0], value[1] };
        const auto s1 = segment<T, 2>{ value[1], value[2] };

        return *intersection(make_line(perpendicular(s0, center(s0))), make_line(perpendicular(s1, center(s1))), epsilon);
    }
};

static constexpr inline auto circumcenter = circumcenter_fn{};

struct incenter_fn
{
    template <typename T>
    constexpr auto operator()(const triangle<T, 2>& value) const -> vector<T, 2>
    {
        T sum = T(0);

        vector<T, 2> result;

        for (size_t i = 0; i < 3; ++i)
        {
            const auto side_length = length(segment<T, 2>{ value[(i + 0) % 3], value[(i + 1) % 3] });

            result += side_length * value[(i + 2) % 3];

            sum += side_length;
        }

        return result / sum;
    }
};

static constexpr inline auto incenter = incenter_fn{};

struct incircle_fn
{
    template <class T>
    constexpr auto operator()(const triangle<T, 2>& triangle) const -> spherical_shape<T, 2>
    {
        constexpr T epsilon = T(0.1);

        const auto center = incenter(triangle);
        const auto radius = distance(center, *projection(center, segment<T, 2>{ triangle[0], triangle[1] }, epsilon));

        return spherical_shape<T, 2>{ center, radius };
    }
};

static constexpr inline auto incircle = incircle_fn{};

struct circumcircle_fn
{
    template <class T>
    constexpr auto operator()(const triangle<T, 2>& triangle) const -> spherical_shape<T, 2>
    {
        const auto center = circumcenter(triangle);
        const auto radius = distance(center, triangle[0]);

        return spherical_shape<T, 2>{ center, radius };
    }
};

static constexpr inline auto circumcircle = circumcircle_fn{};

}  // namespace detail

using detail::altitude;
using detail::angle;
using detail::center;
using detail::centroid;
using detail::circumcenter;
using detail::circumcircle;
using detail::contains;
using detail::cross;
using detail::distance;
using detail::dot;
using detail::incenter;
using detail::incircle;
using detail::interpolate;
using detail::intersection;
using detail::intersects;
using detail::length;
using detail::lower;
using detail::max;
using detail::min;
using detail::norm;
using detail::orthocenter;
using detail::perpendicular;
using detail::projection;
using detail::rejection;
using detail::size;
using detail::unit;
using detail::upper;

template <class T>
using vector_2d = vector<T, 2>;

template <class T>
using vector_3d = vector<T, 3>;

template <class T>
using square_matrix_2d = square_matrix<T, 3>;

template <class T>
using square_matrix_3d = square_matrix<T, 4>;

template <class T>
using region_2d = box_shape<T, 2>;

template <class T>
using region_3d = box_shape<T, 3>;

template <class T>
using rect = region_2d<T>;

template <class T>
using rect_2d = rect<T>;

template <class T>
using cuboid = region_3d<T>;

template <class T>
using cuboid_3d = cuboid<T>;

template <class T>
using line_2d = line<T, 2>;

template <class T>
using ray_2d = ray<T, 2>;

template <class T>
using segment_2d = segment<T, 2>;

template <class T>
using line_3d = line<T, 3>;

template <class T>
using ray_3d = ray<T, 3>;

template <class T>
using segment_3d = segment<T, 3>;

template <class T>
using triangle_2d = triangle<T, 2>;

template <class T>
using quad_2d = quad<T, 2>;

template <class T>
using polygon_2d = polygon<T, 2>;

template <class T>
using polygon_3d = polygon<T, 3>;

template <class T>
using polyline_2d = polyline<T, 2>;

template <class T>
using polyline_3d = polyline<T, 3>;

template <class T>
using circle = spherical_shape<T, 2>;

template <class T>
using circle_2d = circle<T>;

template <class T>
using sphere = spherical_shape<T, 3>;

template <class T>
using sphere_3d = sphere<T>;

}  // namespace mx
