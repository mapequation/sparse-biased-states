#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <clustering.h>

TEST_CASE("js_distance") {
    std::vector<double> p{0.5, 0.5};
    std::vector<double> q{0.5, 0.5};

    CHECK_EQ(js_distance(p, q), 0);
    CHECK_EQ(js_distance(q, p), 0);

    p = {1.0, 0.0};
    q = {0.0, 1.0};
    CHECK_EQ(js_distance(p, q), 1.0);
    CHECK_EQ(js_distance(q, p), 1.0);

    p = {1.0, 1.0, 0};
    q = {0, 1.0, 1.0};
    CHECK_GT(js_distance(p, q), 0.7071);
    CHECK_LT(js_distance(p, q), 0.7072);
    CHECK_EQ(js_distance(p, q), js_distance(q, p));

    p = {1.0, 1.0, 0, 1.0, 1.0, 1.0};
    q = {0, 1.0, 1.0, 10, 1.0, 1.0};
    CHECK_EQ(js_distance(p, q), js_distance(q, p));
}
