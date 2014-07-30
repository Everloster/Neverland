#include <stdio.h>
#include <stdlib.h>
;
/* The ENVIRON variable contains the environment. */
extern char** environ;

// int main(int argc, char const *argv[])
// {
	
// 	return 0;
// }

void printEnv()
{
	/* code */
	char** var;
	for (var = environ; *var != NULL; ++var)
		printf ("%s\n", *var);
}