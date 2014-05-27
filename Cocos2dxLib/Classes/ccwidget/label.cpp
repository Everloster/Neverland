/******************************************************************************
文件名: label.cpp
编写者: csdn_viva@foxmail.com ccwidget1.0
编写日期: 2013-6-5
功能描述: 标签类实现。
******************************************************************************/
#include "widget.h"
#include "label.h"

///静态构造函数，通过制定字体
CLabel* CLabel::create()
{
	return CLabel::create("");
}

///静态构造函数，通过指定文字 文字大小
CLabel* CLabel::create(const char *pString, int nFontSize)
{
	return CLabel::create(pString, LABEL_FONTNAME, nFontSize);
}

///静态构造函数，通过指定文字 文字大小 文字字体来构造
CLabel* CLabel::create(const char *pString, const char* pFontName/*LABEL_FONTNAME*/, int nFontSize/*LABEL_FONTSIZE*/)
{
	CLabel *pRet = new CLabel();
	//默认文字从左上角开始
    if(pRet && pRet->initWithString(pString, pFontName, nFontSize, CCSizeZero, kCCTextAlignmentLeft, kCCVerticalTextAlignmentTop))
    {
        pRet->autorelease();
        return pRet;
    }
    CC_SAFE_DELETE(pRet);
    return NULL;
}
