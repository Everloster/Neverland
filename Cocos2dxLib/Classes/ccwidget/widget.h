/******************************************************************************
文件名: widget.h
编写者: csdn_viva@foxmail.com ccwidget1.0
编写日期: 2013-6-24
功能描述: 包含了所有控件头文件的定义
******************************************************************************/
#ifndef __CCWIDGET_WIDGET_H__
#define __CCWIDGET_WIDGET_H__

#include "cocos2d.h"
#include "cocos-ext.h"
using namespace cocos2d;
using namespace cocos2d::extension;

class CLabel;
class CButton;
class CTableView;
class CTableViewCell;
class CToggleButton;
class CToggleGroup;
class CWidgetLayout;

#define LABEL_FONTNAME "Helvetica"			//Label默认的字体
#define LABEL_FONTSIZE 20					//Label默认的字号
#define BUTTON_FONTNAME "Helvetica"			//Button默认的字体
#define BUTTON_FONTSIZE 20					//Button默认的字号
#define BUTTON_FONTCOLOR ccc3(255,255,255)	//Button默认的字体颜色

#include "label.h"
#include "button.h"
#include "tableview.h"
#include "widgetlayout.h"
#include "togglebutton.h"


#endif //__CCWIDGET_WIDGET_H__