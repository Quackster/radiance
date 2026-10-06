#include "rad_test.h"
#include "../arch/common/rad_arch.h"

using rad::arch::Geom;
using rad::arch::yarn_from;

TEST(yarn_metadata_is_independent_of_deployment_context) {
    const char* keys[] = {"rope_parameters.rope_type", "rope_parameters.factor",
        "rope_parameters.original_max_position_embeddings", "draft.rope_parameters.rope_type",
        "draft.rope_parameters.factor", "draft.rope_parameters.original_max_position_embeddings"};
    const char* vals[] = {"yarn", "2", "262144", "yarn", "2", "262144"};
    RadModelMeta m{}; m.n_kv = 6; m.kv_key = keys; m.kv_val = vals;
    Geom target{}, draft{}; target.theta = draft.theta = 1e7f;
    target.max_ctx = draft.max_ctx = 524288;
    CHECK_OK(yarn_from(target, &m, "", 64));
    CHECK_OK(yarn_from(draft, &m, "draft.", 128));
    CHECK_EQ(target.yarn.low, 14.0f); CHECK_EQ(target.yarn.high, 22.0f);
    CHECK_EQ(draft.yarn.low, 28.0f); CHECK_EQ(draft.yarn.high, 43.0f);
    CHECK_NEAR(target.yarn.magnitude, 1.069314718, 1e-7);
    target.max_ctx = 1048576;
    CHECK_OK(yarn_from(target, &m, "", 64));
    CHECK_EQ(target.yarn.low, 14.0f);
    vals[4] = "4";
    CHECK_OK(yarn_from(draft, &m, "draft.", 128));
    CHECK_EQ(draft.yarn.factor, 4.0f); CHECK_EQ(target.yarn.factor, 2.0f);
}

TEST(yarn_metadata_rejects_bad_inputs_and_preserves_default) {
    const char* keys[] = {"rope_parameters.rope_type", "rope_parameters.factor",
                         "rope_parameters.original_max_position_embeddings"};
    const char* vals[] = {"yarn", "2", "262144"};
    RadModelMeta m{}; m.n_kv = 3; m.kv_key = keys; m.kv_val = vals;
    Geom g{}; g.theta = 1e7f;
    for (const char* bad : {"0", "-1", "nan", "inf", "bogus", "2garbage"}) {
        vals[1] = bad;
        CHECK(yarn_from(g, &m, "", 64) != RAD_OK);
    }
    vals[1] = "2"; m.n_kv = 2;
    CHECK(yarn_from(g, &m, "", 64) != RAD_OK);
    vals[0] = "dynamic";
    CHECK_EQ(yarn_from(g, &m, "", 64), RAD_E_UNSUPPORTED);
    vals[0] = "default"; g.rope_scale = 1.0f;
    CHECK_OK(yarn_from(g, &m, "", 64));
    CHECK_EQ(g.yarn.factor, 0.0f); CHECK_EQ(g.rope_scale, 1.0f);
    vals[0] = "linear";
    CHECK_OK(yarn_from(g, &m, "", 64)); CHECK_EQ(g.rope_scale, 2.0f);
}

TEST(yarn_legacy_explicit_magnitude_and_fractional_bounds) {
    const char* keys[] = {"rope_scaling.type", "rope_scaling.factor",
        "rope_scaling.original_max_position_embeddings", "rope_scaling.attention_factor",
        "rope_scaling.truncate", "rope_scaling.beta_fast", "rope_scaling.beta_slow"};
    const char* vals[] = {"yarn", "2", "262144", "1.25", "0", "32", "1"};
    RadModelMeta m{}; m.n_kv = 7; m.kv_key = keys; m.kv_val = vals;
    Geom g{}; g.theta = 1e7f;
    CHECK_OK(yarn_from(g, &m, "", 64));
    CHECK_EQ(g.yarn.magnitude, 1.25f);
    CHECK(g.yarn.low > 14.0f && g.yarn.low < 15.0f);
    CHECK(g.yarn.high > 21.0f && g.yarn.high < 22.0f);
    vals[4] = "2"; CHECK(yarn_from(g, &m, "", 64) != RAD_OK);
    vals[4] = "0"; vals[2] = "262144.5";
    CHECK(yarn_from(g, &m, "", 64) != RAD_OK);
    vals[2] = "262144"; vals[5] = "0.5";
    CHECK(yarn_from(g, &m, "", 64) != RAD_OK);
}

RAD_TEST_MAIN()
