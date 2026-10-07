#include "rad_test.h"
#include "../arch/common/rad_arch.h"
#include "format/rope_override.h"

#include <cmath>
#include <string>
#include <vector>

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

TEST(linear_metadata_rejects_fp32_overflow_and_underflow) {
    const char* keys[] = {"rope_parameters.rope_type", "rope_parameters.factor"};
    const char* vals[] = {"linear", "2"};
    RadModelMeta m{}; m.n_kv = 2; m.kv_key = keys; m.kv_val = vals;
    Geom g{};
    for (const char* bad : {"1e300", "1e-300", "0", "-2", "nan", "inf"}) {
        vals[1] = bad;
        g.rope_scale = 3.0f;
        CHECK_EQ(yarn_from(g, &m, "", 64), RAD_E_INVAL);
        CHECK_EQ(g.rope_scale, 3.0f);
    }
    vals[1] = "0.5";
    CHECK_OK(yarn_from(g, &m, "", 64));
    CHECK_EQ(g.rope_scale, 0.5f);
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

/* --rope-scaling (core/format/rope_override.h) rewrites a container's keys at load; yarn_from then
 * reads the result as it reads a converted container. These hold the two together. */
namespace {
struct Meta {
    std::vector<std::string> k, v;
    std::vector<const char*> kp, vp;
    RadModelMeta m{};
    Meta(std::initializer_list<std::pair<const char*, const char*>> kvs) {
        for (const auto& p : kvs) { k.push_back(p.first); v.push_back(p.second); }
    }
    const RadModelMeta* meta() {
        kp.clear(); vp.clear();
        for (size_t i = 0; i < k.size(); ++i) { kp.push_back(k[i].c_str()); vp.push_back(v[i].c_str()); }
        m = RadModelMeta{};
        m.n_kv = (int)k.size(); m.kv_key = kp.data(); m.kv_val = vp.data();
        return &m;
    }
    const std::string* get(const std::string& key) const {
        for (size_t i = 0; i < k.size(); ++i) if (k[i] == key) return &v[i];
        return nullptr;
    }
};
/* What a Qwen3.8-Flash-Next container says about its rotary (rad-info --meta). */
Meta qfn_container() {
    return Meta{ { "max_position_embeddings", "262144" },
                 { "rope_parameters.mrope_section", "11 11 10" },
                 { "rope_parameters.partial_rotary_factor", "0.25" },
                 { "rope_parameters.rope_theta", "10000000" },
                 { "rope_parameters.rope_type", "default" },
                 { "text_config.max_position_embeddings", "262144" },
                 { "text_config.rope_parameters.mrope_section", "11 11 10" },
                 { "text_config.rope_parameters.partial_rotary_factor", "0.25" },
                 { "text_config.rope_parameters.rope_theta", "10000000" },
                 { "text_config.rope_parameters.rope_type", "default" } };
}
rad::RopeSpec spec(const char* s) {
    rad::RopeSpec r;
    CHECK(rad::rope_spec_parse(s, &r));
    return r;
}
}  /* namespace */

TEST(rope_spec_parse_takes_yarn_with_a_factor_above_one_or_none) {
    rad::RopeSpec r;
    CHECK(rad::rope_spec_parse("yarn:2", &r));
    CHECK(r.kind == rad::RopeSpec::Yarn); CHECK_EQ(r.factor, 2.0); CHECK_EQ(r.orig, (int64_t)0);
    CHECK(rad::rope_spec_parse("yarn:4:131072", &r));
    CHECK_EQ(r.factor, 4.0); CHECK_EQ(r.orig, (int64_t)131072);
    CHECK(rad::rope_spec_parse("none", &r)); CHECK(r.kind == rad::RopeSpec::None);
    CHECK(rad::rope_spec_parse("", &r)); CHECK(r.kind == rad::RopeSpec::Keep);
    for (const char* bad : { "yarn:1", "yarn:0.5", "yarn:4x", "yarn:", "yarn:4:", "yarn:4:0",
                             "yarn:4:12k", "linear:2", "yarn", "yarn:nan", "yarn:inf" })
        CHECK(!rad::rope_spec_parse(bad, &r));
}

/* THE TRAP THE OVERRIDE EXISTS TO AVOID: the container names its rotary under text_config., and
 * the reader takes the first object that names one -- so YaRN written beside it is never read. */
TEST(rope_override_writes_where_the_reader_looks_on_a_qwen38_container) {
    {
        Meta naive = qfn_container();
        naive.v[4] = "yarn";
        naive.k.push_back("rope_parameters.factor"); naive.v.push_back("2");
        naive.k.push_back("rope_parameters.original_max_position_embeddings");
        naive.v.push_back("262144");
        Geom g{}; g.theta = 1e7f;
        CHECK_OK(yarn_from(g, naive.meta(), "", 64));
        CHECK_EQ(g.yarn.factor, 0.0f);   /* text_config's "default" won */
    }
    Meta m = qfn_container();
    std::string why;
    CHECK(rad::rope_override_group(m.k, m.v, spec("yarn:2"), "", 262144, &why));
    CHECK_EQ(*m.get("text_config.rope_parameters.rope_type"), std::string("yarn"));
    CHECK_EQ(*m.get("rope_parameters.rope_type"), std::string("default"));   /* untouched */
    Geom g{}; g.theta = 1e7f;
    CHECK_OK(yarn_from(g, m.meta(), "", 64));
    CHECK_EQ(g.yarn.factor, 2.0f);
    CHECK_EQ(g.yarn.low, 14.0f); CHECK_EQ(g.yarn.high, 22.0f);
    CHECK_NEAR(g.yarn.magnitude, 1.0 + 0.1 * std::log(2.0), 1e-6);
    CHECK_EQ(g.rope_scale, 1.0f);
    const rad::RopeResolved r = rad::rope_resolve(m.k, m.v, "");
    CHECK_EQ(r.type, std::string("yarn")); CHECK_EQ(r.factor, 2.0); CHECK_EQ(r.orig, 262144.0);
    /* An explicit ORIG is the one used. */
    Meta m2 = qfn_container();
    CHECK(rad::rope_override_group(m2.k, m2.v, spec("yarn:4:131072"), "", 262144, &why));
    Geom g2{}; g2.theta = 1e7f;
    CHECK_OK(yarn_from(g2, m2.meta(), "", 64));
    CHECK_EQ(g2.yarn.factor, 4.0f);
    CHECK(g2.yarn.low < 14.0f);   /* a shorter trained length moves the ramp down */
}

/* A checkpoint's own YaRN: the override's factor replaces it, its trained length stays, and what
 * belonged to the old factor (an explicit attention_factor) goes so the reader derives it anew.
 * `none` takes it off. */
TEST(rope_override_over_a_checkpoints_own_yarn_and_none_takes_it_off) {
    auto legacy = [] {
        return Meta{ { "rope_scaling.type", "yarn" }, { "rope_scaling.factor", "4" },
                     { "rope_scaling.original_max_position_embeddings", "32768" },
                     { "rope_scaling.attention_factor", "1.25" } };
    };
    Meta m = legacy();
    std::string why;
    CHECK(rad::rope_override_group(m.k, m.v, spec("yarn:2"), "", 131072, &why));
    CHECK_EQ(*m.get("rope_scaling.original_max_position_embeddings"), std::string("32768"));
    CHECK(!m.get("rope_scaling.attention_factor"));
    Geom g{}; g.theta = 1e6f;
    CHECK_OK(yarn_from(g, m.meta(), "", 128));
    CHECK_EQ(g.yarn.factor, 2.0f);
    CHECK_NEAR(g.yarn.magnitude, 1.0 + 0.1 * std::log(2.0), 1e-6);
    Meta off = legacy();
    CHECK(rad::rope_override_group(off.k, off.v, spec("none"), "", 131072, &why));
    Geom g3{}; g3.theta = 1e6f; g3.rope_scale = 1.0f;
    CHECK_OK(yarn_from(g3, off.meta(), "", 128));
    CHECK_EQ(g3.yarn.factor, 0.0f);
    CHECK_EQ(g3.rope_scale, 1.0f);
    /* none on a model without YaRN changes nothing. */
    Meta q = qfn_container();
    const std::vector<std::string> before = q.v;
    CHECK(rad::rope_override_group(q.k, q.v, spec("none"), "", 262144, &why));
    CHECK(q.v == before);
}

/* Not stacked over another rotary scheme, and not without a trained length to scale from. The
 * drafter is its own group, over its own trained length. */
TEST(rope_override_refuses_linear_and_scales_a_drafter_on_its_own) {
    Meta lin{ { "rope_parameters.rope_type", "linear" }, { "rope_parameters.factor", "2" } };
    std::string why;
    CHECK(!rad::rope_override_group(lin.k, lin.v, spec("yarn:2"), "", 262144, &why));
    CHECK(why.find("linear") != std::string::npos);
    Meta bare{ { "rope_theta", "1000000" } };
    CHECK(!rad::rope_override_group(bare.k, bare.v, spec("yarn:2"), "", 0, &why));
    CHECK(rad::rope_override_group(bare.k, bare.v, spec("yarn:2:65536"), "", 0, &why));
    CHECK_EQ(*bare.get("rope_parameters.rope_type"), std::string("yarn"));

    Meta d = qfn_container();
    d.k.push_back("draft.rope_parameters.rope_type"); d.v.push_back("default");
    d.k.push_back("draft.max_position_embeddings"); d.v.push_back("262144");
    CHECK(rad::rope_override_group(d.k, d.v, spec("yarn:2"), "", 262144, &why));
    CHECK(rad::rope_override_group(d.k, d.v, spec("yarn:2"), "draft.", 262144, &why));
    Geom t{}, dr{}; t.theta = dr.theta = 1e7f;
    CHECK_OK(yarn_from(t, d.meta(), "", 64));
    CHECK_OK(yarn_from(dr, d.meta(), "draft.", 128));
    CHECK_EQ(t.yarn.factor, 2.0f); CHECK_EQ(dr.yarn.factor, 2.0f);
    CHECK_EQ(dr.yarn.low, 28.0f); CHECK_EQ(dr.yarn.high, 43.0f);
}

RAD_TEST_MAIN()
