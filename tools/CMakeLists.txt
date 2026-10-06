# tools/ -- rad-info, rad-convert, rad-kbench, rad-tune, and the smaller checkers below.
#
# The rad_add_tool tools link rad_core, and they do that rather than reimplementing anything
# because three of them ARE the engine's startup with a different last step: rad-convert,
# rad-kbench and rad-tune each load the same plugins in the same hierarchy and run the same
# declare phase (spec §4.2, §15, §17). A tool that resolved kernels differently from the engine
# would be a tool whose answers are about a different program.
#
# iface.cpp is the seam onto core/plugin and core/build; opshapes.cpp is the operand sizing
# rad-kbench and rad-tune share, and kfixture.cpp the recorded kernel fixture and its input draw.
# All three are compiled into each tool rather than made a library -- three files against four
# small binaries is not worth a target.

set(RAD_TOOL_SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/iface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/opshapes.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kfixture.cpp)

function(rad_add_tool name src)
  add_executable(${name} ${src} ${RAD_TOOL_SHARED})
  # WHOLE_ARCHIVE, for the same reason as ENABLE_EXPORTS below and it is the other half of it.
  # A static archive yields only the objects that resolve a symbol the executable itself
  # references, and rad-convert never calls rad_issue -- so without it core/runtime/issue.cpp is
  # dropped from the link and the architecture plugin, which DOES call it from its step function,
  # fails to load with "undefined symbol: rad_issue". A host that dlopens a plugin has to present
  # the WHOLE plugin-facing ABI, declare phase and run phase alike, whether or not it uses all of
  # it.
  target_link_libraries(${name} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rad_core>")
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/abi)
  # NO HOME DEFINITION HERE. rad_abi carries RAD_DEFAULT_HOME to everything that links it, the
  # engine and the tools alike, and iface.cpp's rad_home() reads that macro, so there is ONE
  # definition of the default home. A second one derived from the install prefix would put the
  # tools one directory away from core/config.cpp's default, and nothing would show it until an
  # install: "no plugins" reads the same whichever wrong directory it came from.
  #
  # ENABLE_EXPORTS, and it is load-bearing rather than tidy. The declare-phase C ABI
  # (rad_decl_weight, rad_decl_op, rad_declare_logits, ...) is implemented by the HOST and an
  # architecture plugin binds to it at dlopen. Without --export-dynamic those symbols are not in
  # the executable's dynamic table and EVERY architecture plugin fails to load with "undefined
  # symbol: rad_decl_weight". Kernel plugins do not need it -- they only export -- which is why
  # rad-schemas and r4d_selftest work without it.
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ENABLE_EXPORTS ON)
endfunction()

rad_add_tool(rad-info     rad_info.cpp)

# The tokeniser's reference encoder: an independent encode path (tokref.h) that core/text's
# pre-tokeniser compiler is held to. A library rather than a tool source because
# tests/tokenizer_test compares against it too; nothing in core/ links it.
add_library(rad_tokref STATIC ${CMAKE_CURRENT_SOURCE_DIR}/tokref.cpp)
target_link_libraries(rad_tokref PUBLIC rad_core)
target_include_directories(rad_tokref PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core)

# rad-tokcheck needs neither plugins nor the declare phase, so it links the core plainly rather
# than through rad_add_tool's whole-archive link.
add_executable(rad-tokcheck rad_tokcheck.cpp)
target_link_libraries(rad-tokcheck PRIVATE rad_tokref rad_core)
set_target_properties(rad-tokcheck PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(rad-tokenize rad_tokenize.cpp)
target_link_libraries(rad-tokenize PRIVATE rad_core)
target_include_directories(rad-tokenize PRIVATE ${CMAKE_SOURCE_DIR}/core)
set_target_properties(rad-tokenize PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Not a rad_add_tool: it links no core at all, deliberately. It dlopens plugins and reads their
# static schema tables, so it works on a plugin the engine cannot even load -- which is exactly
# when you need it.
add_executable(rad-schemas rad_schemas.cpp)
target_link_libraries(rad-schemas PRIVATE rad_abi ${CMAKE_DL_LIBS})
set_target_properties(rad-schemas PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
rad_add_tool(rad-convert  rad_convert.cpp)
# OpenMP across weights, and within a large one. Without it the same code runs one thread.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(rad-convert PRIVATE OpenMP::OpenMP_CXX)
endif()
rad_add_tool(rad-kbench    rad_kbench.cpp)
rad_add_tool(rad-tune     rad_tune.cpp)

# The GBNF mask engine's acceptance run: the differential and latency suite of tests/gbnf_test.cpp
# against a real tokenizer.json and chat template, given by path. Not a rad_add_tool -- it loads no
# plugin and declares nothing -- and not installed: it compiles in upstream's candidate walk as its
# oracle, which is test code (tests/gbnf_oracle.cpp) and belongs in no shipped binary.
add_executable(rad-gbnf-check rad_gbnf_check.cpp ${CMAKE_SOURCE_DIR}/tests/gbnf_oracle.cpp)
target_link_libraries(rad-gbnf-check PRIVATE rad_core)
target_include_directories(rad-gbnf-check PRIVATE
  ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi ${CMAKE_SOURCE_DIR}/tests)
set_target_properties(rad-gbnf-check PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# rad-schemas is the tool an installed prefix needs most: it links no core, so it reads a plugin's
# static schema tables on a plugin the engine cannot even load -- which is exactly the position
# somebody debugging their own out-of-tree plugin is in. The four exports it needs are the four
# every kernel plugin has, so it answers before anything else in the stack will.
install(TARGETS rad-info rad-convert rad-kbench rad-tune rad-schemas rad-tokcheck
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# rad-chatparse: the reply parser's acceptance run against a checkpoint's real tokenizer and chat
# template, beside the legacy parser in tests/legacy/. It shares the differential harness and that
# parser with tests/chatparse_test, so it exists only in a build that has the tests.
if(RAD_BUILD_TESTS)
  add_executable(rad-chatparse rad_chatparse.cpp)
  target_link_libraries(rad-chatparse PRIVATE rad_legacy_toolparse)
  target_include_directories(rad-chatparse PRIVATE ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi)
  set_target_properties(rad-chatparse PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
