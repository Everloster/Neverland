/*
 * IntegerConverter.cpp
 *
 *  Created on: 2014-4-8
 *      Author: 80054369
 */
#include "IntegerConverter.h"

int convert4BytesToIntByBitOption(unsigned char* bytes) {
	int resultIntChar = 0;
	int count = 4;
	while (count > 0) {
		bytes++;
		count--;
	}
	bytes--;
	for (int i = 0; i < 4; i++) {
		resultIntChar <<= 8;
		resultIntChar |= (*bytes & 0xff);
		bytes--;
	}
	return resultIntChar;
}

int convert4BytesToIntByMemCopy(unsigned char* bytes) {
	unsigned int convertInt = 0;
	unsigned char* resultIntChar = bytes;
	unsigned char* convertIntChar = (unsigned char*) &convertInt;
	int count = 4;
	while (count > 0) {
		*convertIntChar = *resultIntChar;
		resultIntChar++;
		convertIntChar++;
		count--;
	}
	return convertInt;
}

bool isCPUBigEndian() {
	unsigned short test = 0x1234;
	if (*((unsigned char*) &test) == 0x12)
//		printlnString("The cpu's endian is big.");
		return true;
	else if (*((unsigned char*) &test) == 0x34)
//		printlnString("The cpu's endian is little");
		return false;
	// what fucked happen.
	return true;
}
