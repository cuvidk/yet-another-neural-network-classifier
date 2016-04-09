QT += core
QT -= gui

CONFIG += c++11

TARGET = artificial-neural-nets
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    neuralnetworkloader.cpp \
    neuralnetwork.cpp \
    fileformatexception.cpp

unix|win32: LIBS += -larmadillo

HEADERS += \
    neuralnetworkloader.h \
    fileopenexception.h \
    fileformatexception.h \
    neuralnetwork.h \
    invalidinputexception.h \
    nnfiletype.h
