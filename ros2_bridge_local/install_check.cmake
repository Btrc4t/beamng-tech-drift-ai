# Verifies the bridge landed in BNG_HOME. This is a configure_file template:
# CMakeLists.txt configures a copy into the build tree with @CHECK_FILE@
# substituted, and the build runs 'cmake -P <build tree>/install_check.cmake'.
# (The path is embedded rather than passed with -D, because the
# ExternalProject Makefile generator mangles quoted -D= arguments.)
if(NOT EXISTS "@CHECK_FILE@")
  message(FATAL_ERROR
    "@CHECK_FILE@ does not exist; the install into BNG_HOME failed "
    "(is BinLinux/ writable?)")
endif()
