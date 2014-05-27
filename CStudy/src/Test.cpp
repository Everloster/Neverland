//============================================================================
// Name        : Test.cpp
// Author      : jabe
// Version     :
// Copyright   : for study
// Description : Hello World in C, Ansi-style
//============================================================================

#include "Test.h"
#include <pthread.h>
#include <unistd.h>

using namespace std;

pthread_key_t key;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond =   PTHREAD_COND_INITIALIZER;
unsigned long abc=0;

int main(void) {

//	testIntStringConvert();

//	testEnum();

	testEndian();

	testEndianConvert();

	return EXIT_SUCCESS;
}


void testIntStringConvert() {
	string message = "hehe";
	printlnString(message.c_str());
	string all_nines(9, '9');
	string lengthMessage;
	lengthMessage.append("the string length is ");
	lengthMessage.append(convertIntToString(all_nines.size()).c_str());
	printlnString(lengthMessage.c_str());
	const char* lengthResultConstChar = lengthMessage.c_str();
	while (*lengthResultConstChar) {
		cout << *lengthResultConstChar << endl;
		lengthResultConstChar++;
	}
}

void testEnum() {
	enum Point {
		point1, point2, point3
	};
	string result = convertIntToString(point1);
	printlnString(result.c_str());
}

void testEndian() {
	if (isCPUBigEndian())
		printlnString("The cpu's endian is big.");
	else if (!isCPUBigEndian())
		printlnString("The cpu's endian is little");
}

void testEndianConvert() {
	printlnString("The size of int is " + convertIntToString(sizeof(int)));
	unsigned int resultInt = 0x12345678;
	unsigned char* resultIntChar = (unsigned char*) &resultInt;
	printlnString("The correct result int is : " + convertIntToString(resultInt));
	printlnString("The bytes copy way to convert int : " + convertIntToString(convert4BytesToIntByMemCopy(resultIntChar)));
	printlnString("The bits add way to convert int : " + convertIntToString(convert4BytesToIntByBitOption(resultIntChar)));
}
