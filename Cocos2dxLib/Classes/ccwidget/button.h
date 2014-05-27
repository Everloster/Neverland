/******************************************************************************
文件名: button.h
编写者: csdn_viva@foxmail.com ccwidget1.0
编写日期: 2013-6-24
功能描述: 按钮定义。
******************************************************************************/
#ifndef __CCWIDGET_BUTTON_H__
#define __CCWIDGET_BUTTON_H__

/******************************************************************************
类    名: CButton
功能描述: 按钮类，继承于CCMenuItem，提供处理事件接口，但是这个按钮实现，不能
		  自己单独处理事件，必须与Layer结合
******************************************************************************/
class CButton : public CCMenuItemImage
{
public:
	///初始化数据
	CButton() : m_nIdx(0), m_pLabel(NULL) { };

public:
	///静态构造函数，重载 creates a menu item with a normal and selected image
    static CButton* create(const char *normalImage, const char *selectedImage);
	///静态构造函数，重载 creates a menu item with a normal,selected  and disabled image
    static CButton* create(const char *normalImage, const char *selectedImage, const char *disabledImage);
	///静态构造函数，重载 creates a menu item with a normal and selected image with target/selector
    static CButton* create(const char *normalImage, const char *selectedImage, CCObject* target, SEL_MenuHandler selector);
	///静态构造函数，重载 creates a menu item with a normal,selected  and disabled image with target/selector
    static CButton* create(const char *normalImage, const char *selectedImage, const char *disabledImage, CCObject* target, SEL_MenuHandler selector);

public:
	///获得下标ID
	virtual int getIdx(){ return m_nIdx; }
	///设置下标ID
	virtual void setIdx(int idx){ m_nIdx = idx; }

	///设置按钮文字
	virtual void setString(const char* pString);
	///设置按钮文字字体
	virtual void setFontName(const char* pFontName);
	///设置按钮文字字号
	virtual void setFontSize(int nFontSize);
	///设置按钮文字颜色
	virtual void setFontColor(const ccColor3B &color);

public:
	///舍去的方法，暂时不支持
	virtual void removeFromParent() {}
    ///舍去的方法，暂时不支持
	virtual void removeFromParentAndCleanup(bool cleanup) {}

protected:
	///如果需要按钮有文字，需要创建Label
	virtual void createLabel();

protected:
	///下标ID,用于列表控件中标识
	int m_nIdx;
	///按钮文字
	CCLabelTTF *m_pLabel;

};


#endif //__CCWIDGET_BUTTON_H__