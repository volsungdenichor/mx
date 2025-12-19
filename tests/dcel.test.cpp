#include <gmock/gmock.h>

#include <mx/dcel.hpp>

template <class T>
struct to_vector
{
    struct reducer_t
    {
        template <class Arg>
        constexpr auto operator()(std::vector<T> state, Arg&& arg) const -> std::vector<T>
        {
            state.push_back(std::forward<Arg>(arg));
            return state;
        }
    };

    reducer_t reducer;
    std::vector<T> state;

    template <class... Args>
    void operator()(Args&&... args)
    {
        state = std::invoke(reducer, std::move(state), std::forward<Args>(args)...);
    }
};

struct vertex_proxy
{
    using type = mx::dcel<float>::vertex_proxy;

    static constexpr auto location = [](auto&& matcher)
    { return testing::Property("location", &type::location, std::forward<decltype(matcher)>(matcher)); };

    static constexpr auto id
        = [](auto&& matcher) { return testing::Field("id", &type::id, std::forward<decltype(matcher)>(matcher)); };
};

struct face_proxy
{
    using type = mx::dcel<float>::face_proxy;

    static constexpr auto id
        = [](auto&& matcher) { return testing::Field("id", &type::id, std::forward<decltype(matcher)>(matcher)); };

    static constexpr auto as_polygon = [](auto&& matcher)
    { return testing::Property("as_polygon", &type::as_polygon, std::forward<decltype(matcher)>(matcher)); };
};

struct halfedge_proxy
{
    using type = mx::dcel<float>::halfedge_proxy;

    static constexpr auto id
        = [](auto&& matcher) { return testing::Field("id", &type::id, std::forward<decltype(matcher)>(matcher)); };

    static constexpr auto vertex_from = [](auto&& matcher)
    { return testing::Property("vertex_from", &type::vertex_from, std::forward<decltype(matcher)>(matcher)); };

    static constexpr auto vertex_to = [](auto&& matcher)
    { return testing::Property("vertex_to", &type::vertex_to, std::forward<decltype(matcher)>(matcher)); };

    static constexpr auto as_segment = [](auto&& matcher)
    { return testing::Property("as_segment", &type::as_segment, std::forward<decltype(matcher)>(matcher)); };

    static constexpr auto incident_face = [](auto&& matcher)
    { return testing::Property("incident_face", &type::incident_face, std::forward<decltype(matcher)>(matcher)); };
};

TEST(dcel, initial_state)
{
    mx::dcel<float> dcel = {};
    EXPECT_THAT(dcel.vertices(to_vector<vertex_proxy::type>{}).state, testing::SizeIs(0));
    EXPECT_THAT(dcel.faces(to_vector<face_proxy::type>{}).state, testing::SizeIs(0));
    EXPECT_THAT(dcel.halfedges(to_vector<halfedge_proxy::type>{}).state, testing::SizeIs(0));
    EXPECT_THROW(dcel.outer_halfedges(to_vector<halfedge_proxy::type>{}).state, std::runtime_error);
}

TEST(dcel, single_vertex)
{
    mx::dcel<float> dcel = {};
    dcel.add_vertex(mx::vector(0.0f, 0.0f));

    EXPECT_THAT(
        dcel.vertices(to_vector<vertex_proxy::type>{}).state,
        testing::ElementsAre(
            testing::AllOf(vertex_proxy::id(testing::Eq(0)), vertex_proxy::location(testing::Eq(mx::vector(0.0f, 0.0f))))));
    EXPECT_THAT(dcel.faces(to_vector<face_proxy::type>{}).state, testing::SizeIs(0));
    EXPECT_THAT(dcel.halfedges(to_vector<halfedge_proxy::type>{}).state, testing::SizeIs(0));
    EXPECT_THROW(dcel.outer_halfedges(to_vector<halfedge_proxy::type>{}).state, std::runtime_error);
}

TEST(dcel, single_face)
{
    mx::dcel<float> dcel = {};
    {
        const auto a = dcel.add_vertex(mx::vector(0.0f, 0.0f));
        const auto b = dcel.add_vertex(mx::vector(2.0f, 0.0f));
        const auto c = dcel.add_vertex(mx::vector(1.0f, 2.0f));
        dcel.add_face({ a, b, c });
    }

    EXPECT_THAT(
        dcel.vertices(to_vector<vertex_proxy::type>{}).state,
        testing::ElementsAre(
            testing::AllOf(vertex_proxy::id(testing::Eq(0)), vertex_proxy::location(testing::Eq(mx::vector(0.0f, 0.0f)))),
            testing::AllOf(vertex_proxy::id(testing::Eq(1)), vertex_proxy::location(testing::Eq(mx::vector(2.0f, 0.0f)))),
            testing::AllOf(vertex_proxy::id(testing::Eq(2)), vertex_proxy::location(testing::Eq(mx::vector(1.0f, 2.0f))))));
    EXPECT_THAT(
        dcel.faces(to_vector<face_proxy::type>{}).state,
        testing::ElementsAre(testing::AllOf(
            face_proxy::id(testing::Eq(0)),
            face_proxy::as_polygon(
                testing::ElementsAre(mx::vector(0.0f, 0.0f), mx::vector(2.0f, 0.0f), mx::vector(1.0f, 2.0f))))));

    EXPECT_THAT(
        dcel.halfedges(to_vector<halfedge_proxy::type>{}).state,
        testing::ElementsAre(
            testing::AllOf(
                halfedge_proxy::id(testing::Eq(0)),
                halfedge_proxy::as_segment(
                    testing::Eq(mx::segment<float, 2>{ mx::vector(0.0f, 0.0f), mx::vector(2.0f, 0.0f) })),
                halfedge_proxy::incident_face(testing::Optional(face_proxy::id(testing::Eq(0)))),
                halfedge_proxy::vertex_from(vertex_proxy::id(testing::Eq(0))),
                halfedge_proxy::vertex_to(vertex_proxy::id(testing::Eq(1)))),
            testing::AllOf(
                halfedge_proxy::id(testing::Eq(1)),
                halfedge_proxy::as_segment(
                    testing::Eq(mx::segment<float, 2>{ mx::vector(2.0f, 0.0f), mx::vector(0.0f, 0.0f) })),
                halfedge_proxy::incident_face(testing::Eq(std::nullopt)),
                halfedge_proxy::vertex_from(vertex_proxy::id(testing::Eq(1))),
                halfedge_proxy::vertex_to(vertex_proxy::id(testing::Eq(0)))),
            testing::AllOf(
                halfedge_proxy::id(testing::Eq(2)),
                halfedge_proxy::as_segment(
                    testing::Eq(mx::segment<float, 2>{ mx::vector(2.0f, 0.0f), mx::vector(1.0f, 2.0f) })),
                halfedge_proxy::incident_face(testing::Optional(face_proxy::id(testing::Eq(0)))),
                halfedge_proxy::vertex_from(vertex_proxy::id(testing::Eq(1))),
                halfedge_proxy::vertex_to(vertex_proxy::id(testing::Eq(2)))),
            testing::AllOf(
                halfedge_proxy::id(testing::Eq(3)),
                halfedge_proxy::as_segment(
                    testing::Eq(mx::segment<float, 2>{ mx::vector(1.0f, 2.0f), mx::vector(2.0f, 0.0f) })),
                halfedge_proxy::incident_face(testing::Eq(std::nullopt)),
                halfedge_proxy::vertex_from(vertex_proxy::id(testing::Eq(2))),
                halfedge_proxy::vertex_to(vertex_proxy::id(testing::Eq(1)))),
            testing::AllOf(
                halfedge_proxy::id(testing::Eq(4)),
                halfedge_proxy::as_segment(
                    testing::Eq(mx::segment<float, 2>{ mx::vector(1.0f, 2.0f), mx::vector(0.0f, 0.0f) })),
                halfedge_proxy::incident_face(testing::Optional(face_proxy::id(testing::Eq(0)))),
                halfedge_proxy::vertex_from(vertex_proxy::id(testing::Eq(2))),
                halfedge_proxy::vertex_to(vertex_proxy::id(testing::Eq(0)))),
            testing::AllOf(
                halfedge_proxy::id(testing::Eq(5)),
                halfedge_proxy::as_segment(
                    testing::Eq(mx::segment<float, 2>{ mx::vector(0.0f, 0.0f), mx::vector(1.0f, 2.0f) })),
                halfedge_proxy::incident_face(testing::Eq(std::nullopt)),
                halfedge_proxy::vertex_from(vertex_proxy::id(testing::Eq(0))),
                halfedge_proxy::vertex_to(vertex_proxy::id(testing::Eq(2))))));

    EXPECT_THROW(dcel.outer_halfedges(to_vector<halfedge_proxy::type>{}).state, std::runtime_error);
}
