QT += core
QT -= gui

TARGET = stereo_matching
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    cpu_src/BM.cpp \
    cpu_src/cost.cpp \
    cpu_src/SGM.cpp \
    cpu_src/Solver.cpp \
    cpu_src/utils.cpp \
    gpu_src/aggregation.cu \
    gpu_src/cost.cu \
    gpu_src/post_filter.cu \
    gpu_src/SGM.cu \
    build/CMakeFiles/3.5.1/CompilerIdCXX/CMakeCXXCompilerId.cpp \
    build/CMakeFiles/feature_tests.cxx \
    build/CMakeFiles/3.5.1/CompilerIdC/CMakeCCompilerId.c \
    build/CMakeFiles/feature_tests.c

HEADERS += \
    cpu_inc/BM.h \
    cpu_inc/cost.h \
    cpu_inc/global.h \
    cpu_inc/SGM.h \
    cpu_inc/Solver.h \
    cpu_inc/utils.h \
    gpu_inc/aggregation.cuh \
    gpu_inc/cost.cuh \
    gpu_inc/post_filter.cuh \
    gpu_inc/SGM.cuh

DISTFILES += \
    CMakeLists.txt

