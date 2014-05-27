/*
 * StringUtils.h
 *
 *  Created on: 2014-2-12
 *      Author: 80054369
 */

#ifndef STRINGUTILS_H_
#define STRINGUTILS_H_
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

void printlnString(const char* message);
void printlnString(string message);
void printlnChar(const char c);

string convertIntToString(const int value);

#endif /* STRINGUTILS_H_ */
