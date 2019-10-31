QT += core
QT -= gui

TARGET = stereo_matching
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += \
    cpu_src/BM.cpp \
    cpu_src/cost.cpp \
    cpu_src/SGM.cpp \
    cpu_src/Solver.cpp \
    cpu_src/utils.cpp \
    sky_detector/imageSkyDetector.cpp \
    demo_old.cpp \
    cpu_src/roshelper.cpp \
    node.cpp

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
    gpu_inc/SGM.cuh \
    sky_detector/imageSkyDetector.h \
    cpu_inc/roshelper.h

DISTFILES += \
    package.xml \
    CMakeLists.txt \
    readme.md \
    gpu_src/aggregation.cu \
    gpu_src/cost.cu \
    gpu_src/post_filter.cu \
    gpu_src/SGM.cu \
    launch/demo.launch

