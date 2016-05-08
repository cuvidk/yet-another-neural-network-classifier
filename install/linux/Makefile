HEADER_DIRS = -I../../source/exception -I../../source/neural_network -I../../source/neural_network_io \
	-I../../source/utils

SOURCES = ../../source/neural_network/neuralnetwork.cpp \
	../../source/neural_network_io/nnio.cpp \
	../../source/exception/fileformatexception.cpp

OBJECTS = neuralnetwork.o nnio.o fileformatexception.o

REQUIRED_LIBS = -larmadillo

OUTPUT_LIB = libyaannc.a
OUTPUT_LIB_DIRECTORY = lib

LIB_INSTALL_PATH = /usr/local/lib
INCLUDE_INSTALL_PATH = /usr/local/include/yaannc
HEADERS_TO_INSTALL = ../../source/neural_network/neuralnetwork.h ../../source/neural_network_io/nnio.h

.SILENT:

all: create-lib clean-objects

install:
	echo --installing lib file in $(LIB_INSTALL_PATH)--
	cp $(OUTPUT_LIB_DIRECTORY)/$(OUTPUT_LIB) $(LIB_INSTALL_PATH)
	echo --creating yaannc directory $(INCLUDE_INSTALL_PATH)--
	mkdir $(INCLUDE_INSTALL_PATH)
	echo --installing required headers under $(INCLUDE_INSTALL_PATH)--
	cp $(HEADERS_TO_INSTALL) $(INCLUDE_INSTALL_PATH)
	echo --installation done, run sudo make uninstall to uninstall the lib--

uninstall:
	echo --removing $(LIB_INSTALL_PATH)/$(OUTPUT_LIB)--
	rm -rf $(LIB_INSTALL_PATH)/$(OUTPUT_LIB)
	echo --removing include directory $(INCLUDE_INSTALL_PATH)--
	rm -rf $(INCLUDE_INSTALL_PATH)
	echo --uninstall complete--

create-lib:
	echo --generating object files-- 
	g++ -c $(SOURCES) $(HEADER_DIRS) $(REQUIRED_LIBS) -std=c++11
	echo --creating $(OUTPUT_LIB_DIRECTORY) directory--
	mkdir $(OUTPUT_LIB_DIRECTORY)
	echo --creating $(OUTPUT_LIB) file--
	ar rc $(OUTPUT_LIB) $(OBJECTS)
	ranlib $(OUTPUT_LIB)
	echo --moving $(OUTPUT_LIB) file inside $(OUTPUT_LIB_DIRECTORY) directory--
	mv $(OUTPUT_LIB) $(OUTPUT_LIB_DIRECTORY)

clean-lib:
	echo --removing $(OUTPUT_LIB_DIRECTORY) directory--
	rm -rf $(OUTPUT_LIB_DIRECTORY)

clean-objects:
	echo --removing object files--
	rm -rf $(OBJECTS)

clean: clean-lib clean-objects

re: clean all
