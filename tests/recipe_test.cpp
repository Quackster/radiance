/* recipe_test.cpp -- the rules rad-convert quantises by (core/format/recipe.h).
 *
 * A recipe decides every weight of a conversion that can take hours, and a rule that silently
 * matched the wrong weights -- or none -- is found only by measuring the container afterwards. So
 * the matching, the order and the two spellings are held here to what spec §4.4 says they are.
 */
#include "rad_test.h"
#include "format/recipe.h"

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

using namespace rad;

TEST(a_glob_matches_the_whole_name) {
    CHECK(recipe_glob("blk.*.ffn_*_exps.*.weight", "blk.12.ffn_down_exps.511.weight"));
    CHECK(recipe_glob("*", "output.weight"));
    CHECK(recipe_glob("output.weight", "output.weight"));
    CHECK(recipe_glob("blk.?.attn_q.weight", "blk.7.attn_q.weight"));
    CHECK(!recipe_glob("blk.?.attn_q.weight", "blk.17.attn_q.weight"));   /* ? is one character */
    CHECK(recipe_glob("blk.*.weight", "blk.1.2.3.weight"));               /* * crosses dots */
    CHECK(!recipe_glob("blk.*.weight", "blk.1.scale"));
    CHECK(!recipe_glob("output", "output.weight"));                        /* not a prefix match */
    CHECK(recipe_glob("*_exps.*", "blk.0.ffn_gate_up_exps.3.weight"));
    CHECK(recipe_glob("a*b*c", "aXXbYYbZc"));                              /* backtracks */
    CHECK(!recipe_glob("a*b*c", "aXXbYY"));
}

TEST(the_first_rule_that_matches_decides) {
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text("blk.*.ffn_down_exps.*.weight  gptq codes=i4 group=128\n"
                          "blk.*.weight                  rtn  codes=fp8_e4m3 block=128x128\n",
                          "r", &why), RAD_OK);
    const RecipeRule* a = r.match("blk.3.ffn_down_exps.7.weight");
    const RecipeRule* b = r.match("blk.3.attn_q.weight");
    REQUIRE(a && b);
    CHECK_EQ(a->quantizer, std::string("gptq"));
    CHECK_EQ(b->quantizer, std::string("rtn"));
    CHECK(r.match("output.weight") == nullptr);                 /* kept as the checkpoint holds it */
}

TEST(a_command_line_rule_goes_ahead_of_the_file) {
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_flag("output.weight=rtn:codes=fp8_e4m3,block=128x128,scale=bf16", &why),
               RAD_OK);
    REQUIRE_EQ(r.add_text("* rtn codes=i8 group=128\n", "file", &why), RAD_OK);
    const RecipeRule* m = r.match("output.weight");
    REQUIRE(m != nullptr);
    CHECK_EQ(m->origin, std::string("--quant"));
    CHECK_EQ(m->options.size(), (size_t)3);
    CHECK_EQ(m->options[0].first, std::string("codes"));
    CHECK_EQ(m->options[2].second, std::string("bf16"));
    CHECK_EQ(r.match("blk.0.attn_q.weight")->origin, std::string("file:1"));
}

TEST(comments_and_blank_lines_are_not_rules_and_a_bad_line_is_named) {
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text("# a recipe\n\n   \nout* rtn codes=i8  # trailing\n", "x.recipe", &why),
               RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)1);
    CHECK_EQ(r.rules()[0].options.size(), (size_t)1);
    CHECK_EQ(r.rules()[0].origin, std::string("x.recipe:4"));

    Recipe bad;
    CHECK_EQ(bad.add_text("a rtn\nlonely\n", "y.recipe", &why), RAD_E_FORMAT);
    CHECK(why.find("y.recipe:2") != std::string::npos);
    CHECK_EQ(bad.add_text("a rtn group\n", "z", &why), RAD_E_FORMAT);   /* not key=value */
    CHECK_EQ(bad.add_text("a rtn k=1 k=2\n", "z", &why), RAD_E_FORMAT); /* given twice */
    CHECK(why.find("twice") != std::string::npos);
    CHECK_EQ(bad.add_flag("noquantiser", &why), RAD_E_FORMAT);
    CHECK_EQ(bad.add_flag("p=:k=v", &why), RAD_E_FORMAT);
}

/* $NAME is the environment's, so one recipe serves every machine -- and an unset name is refused,
 * because a recipe that quantised against an empty calibration directory did something else. */
TEST(a_variable_is_the_environments_and_an_unset_one_is_refused) {
    ::setenv("RAD_RECIPE_TEST_DIR", "/data/calib", 1);
    ::unsetenv("RAD_RECIPE_TEST_UNSET");
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text("* gptq calib=$RAD_RECIPE_TEST_DIR/q38 tag=${RAD_RECIPE_TEST_DIR}x\n",
                          "v", &why), RAD_OK);
    CHECK_EQ(r.rules()[0].options[0].second, std::string("/data/calib/q38"));
    CHECK_EQ(r.rules()[0].options[1].second, std::string("/data/calibx"));
    CHECK_EQ(r.add_text("* gptq calib=$RAD_RECIPE_TEST_UNSET\n", "v", &why), RAD_E_FORMAT);
    CHECK(why.find("RAD_RECIPE_TEST_UNSET") != std::string::npos);
    CHECK_EQ(r.add_text("* gptq calib=${RAD_RECIPE_TEST_DIR\n", "v", &why), RAD_E_FORMAT);
}

/* WHAT A CONTAINER RECORDS: the options in one canonical spelling, so --reuse compares two
 * conversions by what they asked for rather than by how the rule was typed. */
TEST(the_recorded_options_do_not_depend_on_how_they_were_written) {
    Recipe a, b;
    std::string why;
    REQUIRE_EQ(a.add_text("* rtn scale=bf16 codes=i4 group=128\n", "a", &why), RAD_OK);
    REQUIRE_EQ(b.add_flag("*=rtn:group=128,codes=i4,scale=bf16", &why), RAD_OK);
    CHECK_EQ(a.rules()[0].options_text(), b.rules()[0].options_text());
    CHECK_EQ(a.rules()[0].options_text(), std::string("codes=i4,group=128,scale=bf16"));
}

/* The recipe a container carries reads back as the same rules. */
TEST(the_recorded_recipe_reads_back_as_the_same_rules) {
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_flag("output.weight=rtn:codes=fp8_e4m3,block=128x128", &why), RAD_OK);
    REQUIRE_EQ(r.add_text("blk.*.weight gptq codes=i4 group=128 damp=0.02\n", "f", &why), RAD_OK);
    Recipe back;
    REQUIRE_EQ(back.add_text(r.text(), "header", &why), RAD_OK);
    REQUIRE_EQ(back.rules().size(), r.rules().size());
    for (size_t i = 0; i < r.rules().size(); ++i) {
        CHECK_EQ(back.rules()[i].pattern, r.rules()[i].pattern);
        CHECK_EQ(back.rules()[i].quantizer, r.rules()[i].quantizer);
        CHECK(back.rules()[i].options == r.rules()[i].options);
    }
}

namespace {
bool has_opt(const RecipeRule* r, const char* k, const char* v) {
    for (const auto& [kk, vv] : r->options)
        if (kk == k && vv == v) return true;
    return false;
}
}  /* namespace */

/* THE NVFP4 RECIPE, READ FROM THE FILE IT SHIPS IN: it parses, and its patterns decide the real
 * declared names of the 27B architecture -- the fused MLPs and the GDN in-projections (whose a/b
 * pieces ship bf16 beside the NVFP4 ones) to the MXFP4 target, the head to block fp8, the norms
 * and the convolution to the checkpoint's own form. A pattern that matched the wrong weight here
 * would be found only by measuring the container afterwards. */
TEST(the_nvfp4_recipe_decides_the_model_it_was_written_for) {
    std::ifstream in(std::string(RAD_RECIPES) + "/q38-27b-nvfp4-df2.recipe");
    REQUIRE(in.good());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text(text, "q38-27b-nvfp4-df2.recipe", &why), RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)17);

    /* every linear of the model, the whole layer range, to the MXFP4 target -- the fused names
     * the builder hands out, each with its `.weight` suffix */
    const char* linears[] = { "blk.47.ffn_gate_up.weight", "blk.47.ffn_down.weight",
                              "blk.3.ffn_gate_up.weight", "blk.3.attn_qg.weight",
                              "blk.3.attn_k.weight", "blk.3.attn_v.weight",
                              "blk.3.attn_output.weight", "blk.47.ssm_inz.weight",
                              "blk.47.ssm_out.weight" };
    for (const char* n : linears) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("rtn"));
        CHECK(has_opt(m, "codes", "fp4_e2m1"));
        CHECK(has_opt(m, "group", "32"));
        CHECK(has_opt(m, "scale", "e8m0"));
    }
    /* the head to block fp8, not the fp4 of the trunk */
    const RecipeRule* head = r.match("output.weight");
    REQUIRE(head != nullptr);
    CHECK(has_opt(head, "codes", "fp8_e4m3"));
    CHECK(has_opt(head, "block", "128x128"));
    /* a norm, the convolution, the delta net's a/b projection (kept bf16, as the checkpoint
     * ships it) and the embedding keep the checkpoint's own form */
    const char* kept[] = { "blk.0.ffn_norm.weight", "blk.0.attn_norm.weight", "blk.47.ssm_conv1d.weight",
                           "blk.47.ssm_a_log", "blk.47.ssm_ab.weight", "token_embd.weight", "output_norm.weight" };
    for (const char* n : kept)
        CHECK(r.match(n) == nullptr);
    /* the drafter's own linears */
    const RecipeRule* fc = r.match("dflash.fc.weight");
    REQUIRE(fc != nullptr);
    CHECK(has_opt(fc, "codes", "fp4_e2m1"));
    const RecipeRule* dh = r.match("dflash.draft_head.weight");
    REQUIRE(dh != nullptr);
    CHECK(has_opt(dh, "codes", "u2"));
}

RAD_TEST_MAIN()
