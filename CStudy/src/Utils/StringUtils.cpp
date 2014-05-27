#include "StringUtils.h"

void printlnString(const char* message) {
	cout << message << endl;
}

void printlnString(string message) {
	cout << message << endl;
}

void printlnChar(const char c) {
	cout << c << endl;
}

string convertIntToString(const int value) {
	ostringstream convert;
	convert << value;
	return convert.str();
}
