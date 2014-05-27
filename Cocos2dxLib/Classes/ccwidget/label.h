/******************************************************************************
文件名: label.h
编写者: csdn_viva@foxmail.com ccwidget1.0
编写日期: 2013-6-5
功能描述: 标签类定义。
******************************************************************************/
#ifndef __CCWIDGET_LABEL_H__
#define __CCWIDGET_LABEL_H__

/******************************************************************************
类    名: CLabel
功能描述: 标签类，继承与CCLabelTTF实现其他附加的功能
******************************************************************************/
class CLabel : public CCLabelTTF
{
public:
	CLabel(){};
	virtual ~CLabel(){};

public:
	///静态构造函数
	static CLabel* create();
	///静态构造函数，通过指定文字 文字大小
	static CLabel* create(const char *pString, int nFontSize);
	///静态构造函数，通过指定文字 文字大小 文字字体来构造
	static CLabel* create(const char *pString, const char* pFontName = LABEL_FONTNAME, int nFontSize = LABEL_FONTSIZE);
};

#endif // __CCWIDGET_LABEL_H__

