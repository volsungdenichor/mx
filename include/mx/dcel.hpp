#pragma once

#include <map>
#include <mx/mx.hpp>
#include <unordered_map>

namespace mx
{

namespace detail
{

using dcel_vertex_id = int;
using dcel_face_id = int;
using dcel_halfedge_id = int;

struct vertex_info
{
    dcel_vertex_id id;
    dcel_halfedge_id halfedge = dcel_halfedge_id{ -1 };

    friend std::ostream& operator<<(std::ostream& os, const vertex_info& item)
    {
        return os << "V[" << item.id << "] he=" << item.halfedge;
    }
};

struct face_info
{
    dcel_face_id id;
    dcel_halfedge_id halfedge = dcel_halfedge_id{ -1 };

    friend std::ostream& operator<<(std::ostream& os, const face_info& item)
    {
        return os << "F[" << item.id << "] he=" << item.halfedge;
    }
};

struct halfedge_info
{
    dcel_halfedge_id id;
    dcel_vertex_id vertex_from = dcel_vertex_id{ -1 };
    dcel_halfedge_id twin_halfedge = dcel_halfedge_id{ -1 };
    dcel_halfedge_id next_halfedge = dcel_halfedge_id{ -1 };
    dcel_halfedge_id prev_halfedge = dcel_halfedge_id{ -1 };
    dcel_face_id face = dcel_face_id{ -1 };

    friend std::ostream& operator<<(std::ostream& os, const halfedge_info& item)
    {
        return os << "HE[" << item.id << "] from_v=" << item.vertex_from << " twin_he=" << item.twin_halfedge << " "
                  << item.prev_halfedge << ">" << item.id << ">" << item.next_halfedge << " F=" << item.face;
    }
};

template <class T>
class dcel
{
public:
    using location_type = vector<T, 2>;
    using segment_type = segment<T, 2>;
    using polygon_type = polygon<T, 2>;

    struct vertex_proxy;
    struct face_proxy;
    struct halfedge_proxy;

    struct vertex_proxy
    {
        const dcel* m_self;
        dcel_vertex_id id;

        const location_type& location() const
        {
            return m_self->get_location(id);
        }

        template <class Acc>
        Acc out_halfedges(Acc acc) const
        {
            static_assert(std::is_invocable_v<Acc, const halfedge_proxy&>, "(const halfedge_proxy&)");
            auto next = halfedge_proxy{ m_self, info().halfedge };
            const auto first_id = next.id;
            while (true)
            {
                auto current = next;
                next = next.twin_halfedge().next_halfedge();
                acc(current);
                if (next.id == first_id)
                {
                    break;
                }
            }
            return acc;
        }

        template <class Acc>
        Acc in_halfedges(Acc acc) const
        {
            static_assert(std::is_invocable_v<Acc, const halfedge_proxy&>, "(const halfedge_proxy&)");
            auto next = halfedge_proxy{ m_self, info().halfedge };
            const auto first_id = next.id;
            while (true)
            {
                auto current = next;
                next = next.twin_halfedge().next_halfedge();
                acc(current.twin_halfedge());
                if (next.id == first_id)
                {
                    break;
                }
            }
            return acc;
        }

        template <class Acc>
        Acc incident_faces(Acc acc) const
        {
            static_assert(std::is_invocable_v<Acc, const face_proxy&>, "(const face_proxy&)");
            auto next = halfedge_proxy{ m_self, info().halfedge };
            const auto first_id = next.id;
            while (true)
            {
                auto current = next;
                next = next.twin_halfedge().next_halfedge();
                if (auto f = next.incident_face())
                {
                    acc(*f);
                }
                if (next.id == first_id)
                {
                    break;
                }
            }
            return acc;
        }

        friend std::ostream& operator<<(std::ostream& os, const vertex_proxy& item)
        {
            return os << "V: " << item.id << " " << item.location();
        }

        const vertex_info& info() const
        {
            return m_self->get_vertex(id);
        }
    };

    struct face_proxy
    {
        const dcel* m_self;
        dcel_face_id id;

        template <class Acc>
        Acc outer_halfedges(Acc acc) const
        {
            static_assert(std::is_invocable_v<Acc, const halfedge_proxy&>, "(const halfedge_proxy&)");
            auto next = halfedge_proxy{ m_self, info().halfedge };
            const auto first_id = next.id;
            while (true)
            {
                auto current = next;
                next = next.next_halfedge();
                acc(current);
                if (next.id == first_id)
                {
                    break;
                }
            }
            return acc;
        }

        template <class Acc>
        Acc outer_vertices(Acc acc) const
        {
            static_assert(std::is_invocable_v<Acc, const vertex_proxy&>, "(const vertex_proxy&)");
            auto next = halfedge_proxy{ m_self, info().halfedge };
            const auto first_id = next.id;
            while (true)
            {
                auto current = next;
                next = next.next_halfedge();
                acc(current.vertex_from());
                if (next.id == first_id)
                {
                    break;
                }
            }
            return acc;
        }

        template <class Acc>
        Acc adjacent_faces(Acc acc) const
        {
            static_assert(std::is_invocable_v<Acc, const face_proxy&>, "(const face_proxy&)");
            auto next = halfedge_proxy{ m_self, info().halfedge };
            const auto first_id = next.id;
            while (true)
            {
                auto current = next;
                next = next.next.next_halfedge();
                if (auto f = next->twin_halfedge().incident_face())
                {
                    acc(*f);
                }
                if (next.id == first_id)
                {
                    break;
                }
            }
            return acc;
        }

        polygon_type as_polygon() const
        {
            polygon_type out;
            outer_vertices([&](const vertex_proxy& v) { out.push_back(v.location()); });
            return out;
        }

        friend std::ostream& operator<<(std::ostream& os, const face_proxy& item)
        {
            return os << "F " << item.id;
        }

        const face_info& info() const
        {
            return m_self->get_face(id);
        }
    };

    struct halfedge_proxy
    {
        const dcel* m_self;
        dcel_halfedge_id id;

        std::optional<face_proxy> incident_face() const
        {
            const auto& i = info();
            if (i.face == dcel_face_id{ -1 })
            {
                return {};
            }
            return face_proxy{ m_self, i.face };
        }

        halfedge_proxy twin_halfedge() const
        {
            return { m_self, info().twin_halfedge };
        }

        halfedge_proxy next_halfedge() const
        {
            return { m_self, info().next_halfedge };
        }

        halfedge_proxy prev_halfedge() const
        {
            return { m_self, info().prev_halfedge };
        }

        vertex_proxy vertex_from() const
        {
            return { m_self, info().vertex_from };
        }

        vertex_proxy vertex_to() const
        {
            return { m_self, twin_halfedge().vertex_from().id };
        }

        segment_type as_segment() const
        {
            return segment_type{ vertex_from().location(), vertex_to().location() };
        }

        friend std::ostream& operator<<(std::ostream& os, const halfedge_proxy& item)
        {
            return os << item.info();
        }

    private:
        const halfedge_info& info() const
        {
            return m_self->get_halfedge(id);
        }
    };

    std::vector<vertex_info> m_vertices;
    std::vector<location_type> m_locations;
    std::vector<face_info> m_faces;
    std::vector<halfedge_info> m_halfedges;
    std::map<std::pair<dcel_vertex_id, dcel_vertex_id>, dcel_halfedge_id> m_edges;
    dcel_halfedge_id m_boundary_halfedge;

    dcel() : m_vertices{}, m_locations{}, m_faces{}, m_halfedges{}, m_edges{}, m_boundary_halfedge{ -1 }
    {
    }

    dcel_vertex_id add_vertex(const location_type& location)
    {
        vertex_info& v = new_vertex();
        set_location(v.id, location);
        return v.id;
    }

    dcel_face_id add_face(const std::vector<dcel_vertex_id>& vertices)
    {
        if (vertices.size() < 3)
        {
            throw std::runtime_error{ "add_face: at least 3 vertices required" };
        }
        face_info& face = new_face();
        build_face(vertices, &face);
        return face.id;
    }

    void add_boundary()
    {
        m_boundary_halfedge = build_face(hull(), nullptr);
    }

    template <class Acc>
    Acc vertices(Acc acc) const
    {
        static_assert(std::is_invocable_v<Acc, const vertex_proxy&>, "(const vertex_proxy&)");
        for (const vertex_info& v : m_vertices)
        {
            acc(vertex_proxy{ this, v.id });
        }
        return acc;
    }

    template <class Acc>
    Acc faces(Acc acc) const
    {
        static_assert(std::is_invocable_v<Acc, const face_proxy&>, "(const face_proxy&)");
        for (const face_info& f : m_faces)
        {
            acc(face_proxy{ this, f.id });
        }
        return acc;
    }

    template <class Acc>
    Acc halfedges(Acc acc) const
    {
        static_assert(std::is_invocable_v<Acc, const halfedge_proxy&>, "(const halfedge_proxy&)");
        for (const halfedge_info& h : m_halfedges)
        {
            acc(halfedge_proxy{ this, h.id });
        }
        return acc;
    }

    template <class Acc>
    Acc outer_halfedges(Acc acc) const
    {
        static_assert(std::is_invocable_v<Acc, const halfedge_proxy&>, "(const halfedge_proxy&)");
        if (m_boundary_halfedge == dcel_halfedge_id{ -1 })
        {
            throw std::runtime_error{ "boundary not defined " };
        }
        halfedge_proxy next = halfedge_proxy{ this, m_boundary_halfedge };
        const auto first_id = next.id;
        while (true)
        {
            auto current = next;
            acc(current);
            next = next.next_halfedge();
            if (next.id == first_id)
            {
                break;
            }
        }
        return acc;
    }

private:
    const location_type& get_location(dcel_vertex_id id) const
    {
        return m_locations.at(id);
    }

    const vertex_info& get_vertex(dcel_vertex_id id) const
    {
        return m_vertices.at(id);
    }

    vertex_info& get_vertex(dcel_vertex_id id)
    {
        return m_vertices.at(id);
    }

    const halfedge_info& get_halfedge(dcel_halfedge_id id) const
    {
        return m_halfedges.at(id);
    }

    halfedge_info& get_halfedge(dcel_halfedge_id id)
    {
        return m_halfedges.at(id);
    }

    const face_info& get_face(dcel_face_id id) const
    {
        return m_faces.at(id);
    }

    face_info& get_face(dcel_face_id id)
    {
        return m_faces.at(id);
    }

    template <class IdType, class Type, class Func>
    static Type& new_item(std::vector<Type>& container, Func func)
    {
        auto id = IdType{ static_cast<int>(container.size()) };
        container.push_back(std::invoke(func, id));
        return container.back();
    }

    vertex_info& new_vertex()
    {
        return new_item<dcel_vertex_id>(m_vertices, [](dcel_vertex_id id) { return vertex_info{ id }; });
    }

    void set_location(dcel_vertex_id id, const location_type& location)
    {
        m_locations.resize(id + 1);
        m_locations.at(id) = location;
    }

    face_info& new_face()
    {
        return new_item<dcel_face_id>(m_faces, [](dcel_face_id id) { return face_info{ id }; });
    }

    dcel_halfedge_id build_face(const std::vector<dcel_vertex_id>& vertices, face_info* face)
    {
        const auto buffer_begin = std::begin(vertices);
        const auto buffer_end = std::end(vertices);
        const auto buffer_size = static_cast<int>(std::distance(buffer_begin, buffer_end));

        const auto get = [=](int n) -> dcel_vertex_id
        {
            while (n < 0)
            {
                n += buffer_size;
            }

            return *(buffer_begin + (n % buffer_size));
        };

        for (int i = 0; i < buffer_size; ++i)
        {
            connect(get(i + 0), get(i + 1));
        }

        for (int i = 0; i < buffer_size; ++i)
        {
            halfedge_info& h0 = get_halfedge(*find_halfedge(get(i + 0), get(i + 1)));
            halfedge_info& h1 = get_halfedge(*find_halfedge(get(i + 1), get(i + 2)));

            if (vertex_info& v = get_vertex(get(i)); v.halfedge == dcel_halfedge_id{ -1 })
            {
                v.halfedge = h0.id;
            }

            h0.next_halfedge = h1.id;
            h1.prev_halfedge = h0.id;

            if (face)
            {
                if (i == 0)
                {
                    face->halfedge = h0.id;
                }

                h0.face = face->id;
            }
        }

        return m_edges.at(std::pair{ get(0), get(1) });
    }

    std::optional<dcel_halfedge_id> find_halfedge(dcel_vertex_id from, dcel_vertex_id to)
    {
        const auto key = std::pair{ from, to };
        if (const auto iter = m_edges.find(key); iter != m_edges.end())
        {
            return iter->second;
        }
        return {};
    }

    std::pair<dcel_halfedge_id, dcel_halfedge_id> connect(dcel_vertex_id from_vertex, dcel_vertex_id to_vertex)
    {
        const auto f = find_halfedge(from_vertex, to_vertex);
        const auto t = find_halfedge(to_vertex, from_vertex);

        if (f && t)
        {
            return { *f, *t };
        }

        auto [new_halfedges_begin, new_halfedges_end] = add_halfedges(2);

        halfedge_info& from_halfedge = new_halfedges_begin[0];
        halfedge_info& to_halfedge = new_halfedges_begin[1];

        from_halfedge.vertex_from = from_vertex;
        from_halfedge.twin_halfedge = to_halfedge.id;

        to_halfedge.vertex_from = to_vertex;
        to_halfedge.twin_halfedge = from_halfedge.id;

        m_edges.emplace(std::pair{ from_vertex, to_vertex }, from_halfedge.id);
        m_edges.emplace(std::pair{ to_vertex, from_vertex }, to_halfedge.id);

        return { from_halfedge.id, to_halfedge.id };
    }

    std::pair<std::vector<halfedge_info>::iterator, std::vector<halfedge_info>::iterator> add_halfedges(int count)
    {
        for (int i = 0; i < count; ++i)
        {
            new_item<dcel_halfedge_id>(m_halfedges, [](dcel_halfedge_id id) { return halfedge_info{ id }; });
        }
        return { m_halfedges.end() - count, m_halfedges.end() };
    }

    std::vector<dcel_vertex_id> hull() const
    {
        std::vector<dcel_vertex_id> result;

        std::unordered_map<dcel_vertex_id, dcel_vertex_id> outer_halfedges;
        halfedges(
            [&](const halfedge_proxy& h)
            {
                if (!h.incident_face().has_value())
                {
                    outer_halfedges.emplace(h.vertex_from().id, h.vertex_to().id);
                }
            });

        if (outer_halfedges.empty())
        {
            throw std::runtime_error{ "error on creating hull" };
        }

        dcel_vertex_id cur = std::begin(outer_halfedges)->first;

        while (result.size() != outer_halfedges.size())
        {
            dcel_vertex_id n = outer_halfedges.at(cur);
            result.push_back(n);
            cur = n;
        }

        return result;
    }
};

struct voronoi_fn
{
    template <class T>
    dcel<T> operator()(const dcel<T>& input) const
    {
        const std::vector<dcel_vertex_id> outer = std::invoke(
            [&]() -> std::vector<dcel_vertex_id>
            {
                std::vector<dcel_vertex_id> res;
                input.outer_halfedges([&](const typename dcel<T>::halfedge_proxy& halfedge)
                                      { res.push_back(halfedge.vertex_from().id); });

                return res;
            });

        const auto is_outer_vertex = [&](const typename dcel<T>::vertex_proxy& vertex)
        { return std::find(outer.begin(), outer.end(), vertex.id) == outer.end(); };

        std::unordered_map<dcel_face_id, dcel_vertex_id> centers;

        const auto is_outer_face = [&](const typename dcel<T>::face_proxy& face) -> bool
        {
            bool result = false;
            face.outer_halfedges(
                [&](const typename dcel<T>::halfedge_proxy& halfedge)
                { result |= (is_outer_vertex(halfedge.vertex_from()) && is_outer_vertex(halfedge.vertex_to())); });
            return result;
        };

        dcel<T> result;

        const auto add_face = [&](const typename dcel<T>::vertex_proxy& vertex)
        {
            std::vector<dcel_vertex_id> vertices;
            vertex.incident_faces(
                [&](const typename dcel<T>::face_proxy& face)
                {
                    dcel_vertex_id v;
                    if (auto it = centers.find(face.id); it != centers.end())
                    {
                        v = it->second;
                    }
                    else
                    {
                        v = result.add_vertex(get_center<T>(face));
                        centers[face.id] = v;
                    }
                    vertices.push_back(v);
                });

            if (vertices.size() >= 3)
            {
                std::reverse(vertices.begin(), vertices.end());
                result.add_face(vertices);
            }
        };

        input.vertices(
            [&](const typename dcel<T>::vertex_proxy& vertex)
            {
                if (!is_outer_vertex(vertex))
                {
                    add_face(vertex);
                }
            });

        result.add_boundary();

        return result;
    }

    template <class T>
    static triangle<T, 2> as_triangle(const typename dcel<T>::face_proxy& face)
    {
        const auto p = face.as_polygon();
        return { p.at(0), p.at(1), p.at(2) };
    }

    template <class T>
    static vector<T, 2> get_center(const typename dcel<T>::face_proxy& face)
    {
        return circumcenter(as_triangle<T>(face));
    }
};

constexpr inline auto voronoi = voronoi_fn{};

struct triangulate_fn
{
    template <class T>
    dcel<T> operator()(std::vector<vector<T, 2>> vertices) const
    {
        using triangle_info = std::array<std::size_t, 3>;
        using edge_info = std::array<std::size_t, 2>;
        using triangle_type = triangle<T, 2>;

        const auto get_vertex = [&](std::size_t index) -> const vector_2d<T>& { return vertices.at(index); };

        const auto get_triangle = [&](const triangle_info& t) -> triangle_type {
            return { get_vertex(t[0]), get_vertex(t[1]), get_vertex(t[2]) };
        };

        static const auto collinear = [](const triangle_type& t) -> bool
        { return std::abs(orientation(t[0], t[1], t[2]) - 0.0) < std::numeric_limits<T>::epsilon(); };

        const auto bounds = make_aabb(vertices);

        const auto max_dimension = std::max(bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]);

        const auto c = center(bounds);

        const T delta = T(20);

        std::vector<triangle_info> triangles;

        const auto s = vertices.size();

        vertices.push_back(c + vector_2d<T>{ -delta * max_dimension, -max_dimension });
        vertices.push_back(c + vector_2d<T>{ 0, +delta * max_dimension });
        vertices.push_back(c + vector_2d<T>{ +delta * max_dimension, -max_dimension });

        const triangle_info super_triangle{ s + 0, s + 1, s + 2 };

        triangles.push_back(super_triangle);

        for (size_t i = 0; i < vertices.size(); ++i)
        {
            std::vector<triangle_info> invalid_triangles;
            std::vector<edge_info> edges;

            for (const triangle_info& triangle : triangles)
            {
                if (contains(circumcircle(get_triangle(triangle)), get_vertex(i)))
                {
                    invalid_triangles.push_back(triangle);

                    edges.push_back({ triangle.at(0), triangle.at(1) });
                    edges.push_back({ triangle.at(1), triangle.at(2) });
                    edges.push_back({ triangle.at(2), triangle.at(0) });
                }
            }
        }

        triangles.push_back(super_triangle);

        for (size_t i = 0; i < vertices.size(); ++i)
        {
            std::vector<triangle_info> invalid_triangles;
            std::vector<edge_info> edges;

            for (const triangle_info& triangle : triangles)
            {
                if (contains(circumcircle(get_triangle(triangle)), get_vertex(i)))
                {
                    invalid_triangles.push_back(triangle);

                    edges.push_back({ triangle.at(0), triangle.at(1) });
                    edges.push_back({ triangle.at(1), triangle.at(2) });
                    edges.push_back({ triangle.at(2), triangle.at(0) });
                }
            }

            remove(triangles, invalid_triangles);

            std::vector<edge_info> invalid_edges;

            for (const edge_info& edge1 : edges)
            {
                const auto v1 = get_vertices(edge1);
                for (const edge_info& edge2 : edges)
                {
                    if (&edge1 != &edge2 && v1 == get_vertices(edge2))
                    {
                        invalid_edges.push_back(edge1);
                    }
                }
            }

            remove(edges, invalid_edges);

            for (const edge_info& edge : edges)
            {
                triangle_info triangle{ edge[0], edge[1], i };

                if (!collinear(get_triangle(triangle)))
                {
                    triangles.push_back(triangle);
                }
            }
        }

        triangles.erase(
            std::remove_if(
                triangles.begin(),
                triangles.end(),
                [&](const triangle_info& triangle)
                {
                    const triangle_type t = get_triangle(triangle);
                    return std::any_of(
                        super_triangle.begin(),
                        super_triangle.end(),
                        [&](std::size_t super_triangle_vertex) { return contains(t, get_vertex(super_triangle_vertex)); });
                }),
            triangles.end());

        dcel<T> result;

        std::map<size_t, typename dcel<T>::vertex_id> map;

        for (const triangle_info& triangle : triangles)
        {
            for (size_t v : triangle)
            {
                if (map.find(v) == map.end())
                {
                    map[v] = result.add_vertex(get_vertex(v));
                }
            }
        }

        for (triangle_info& triangle : triangles)
        {
            const triangle_2d<T> t = get_triangle(triangle);
            if (cross(t[1] - t[0], t[2] - t[0]) > 0.0)
            {
                std::reverse(triangle.begin(), triangle.end());
            }
            result.add_face({ map.at(triangle[0]), map.at(triangle[1]), map.at(triangle[2]) });
        }

        result.add_boundary();
        return result;
    }

    template <class T>
    static box_shape<T, 2> make_aabb(const std::vector<vector<T, 2>>& vertices)
    {
        box_shape<T, 2> result;
        for (const vector<T, 2>& vertex : vertices)
        {
            result = box_shape<T, 2>{
                interval<T>{ std::min(result[0][0], vertex[0]), std::max(result[0][1], vertex[0]) },
                interval<T>{ std::min(result[1][0], vertex[1]), std::max(result[1][1], vertex[1]) },
            };
        }
        return result;
    }

    template <class T>
    void remove(std::vector<T>& lhs, const std::vector<T>& rhs) const
    {
        lhs.erase(
            std::remove_if(
                lhs.begin(),
                lhs.end(),
                [&](const T& lhs_item)
                {
                    const auto lhs_vertices = get_vertices(lhs_item);
                    return std::any_of(
                        rhs.begin(), rhs.end(), [&](const T& rhs_item) { return lhs_vertices == get_vertices(rhs_item); });
                }),
            lhs.end());
    }

    template <class T>
    std::set<std::size_t> get_vertices(const T& item) const
    {
        return std::set<std::size_t>{ item.begin(), item.end() };
    }
};

constexpr inline auto triangulate = triangulate_fn{};

}  // namespace detail

using detail::dcel;
using detail::triangulate;
using detail::voronoi;

}  // namespace mx
