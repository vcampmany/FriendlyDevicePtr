CC = nvcc
CFLAGS = -std=c++11 -g -G --compiler-options -Wall
DIR = examples/
DEPS = DevicePtr.hpp $(DIR)common.cuh

all: simpleExample interfaceExample
	
simpleExample: $(DIR)SimpleExample.cu $(DEPS)
	$(CC) $(CFLAGS) $(DIR)SimpleExample.cu -o simpleExample	
	
interfaceExample: $(DIR)InterfaceExample.cu $(DEPS)
	$(CC) $(CFLAGS) $(DIR)InterfaceExample.cu -o interfaceExample
	
clean:
	rm simpleExample interfaceExample