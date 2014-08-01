#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
	/* code */

	void* handle = dlopen ("libGetEnvTest.so", RTLD_LAZY);
	void (*test)() = dlsym (handle, "printEnv");
	(*test)();
	dlclose (handle);
	return 0;
}