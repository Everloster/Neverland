/*
 * IntegerConverter.h
 *
 *  Created on: 2014-4-8
 *      Author: 80054369
 */

#ifndef INTEGERCONVERTER_H_
#define INTEGERCONVERTER_H_

int convert4BytesToIntByBitOption(unsigned char* bytes);

int convert4BytesToIntByMemCopy(unsigned char* bytes);

bool isCPUBigEndian();

#endif /* INTEGERCONVERTER_H_ */
