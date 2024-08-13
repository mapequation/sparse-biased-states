cmake_minimum_required(VERSION 3.24)
project(sparse_states)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

file(GLOB project_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB project_HEADERS CONFIGURE_DEPENDS "src/*.h" "src/*.hpp")

add_executable(sparse_states ${project_SOURCES} ${project_HEADERS})

include_directories(src)
add_executable(tests test/tests.cpp ${project_HEADERS})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sparse_states PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(fmt REQUIRED)
if (fmt_FOUND)
    target_link_libraries(sparse_states PRIVATE fmt::fmt-header-only)
endif ()
message("-- Found fmt: ${fmt_FOUND}")

find_program(CCACHE ccache)
if (CCACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
endif ()
message("-- Found Ccache: ${CCACHE}")
