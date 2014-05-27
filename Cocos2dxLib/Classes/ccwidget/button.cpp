/******************************************************************************
文件名: button.cpp
编写者: csdn_viva@foxmail.com ccwidget1.0
编写日期: 2013-6-24
功能描述: 按钮实现。
******************************************************************************/
#include "widget.h"
#include "button.h"

///静态构造函数，重载 creates a menu item with a normal and selected image
CButton* CButton::create(const char *normalImage, const char *selectedImage)
{
    return CButton::create(normalImage, selectedImage, NULL, NULL, NULL);
}

///静态构造函数，重载 creates a menu item with a normal,selected  and disabled image
CButton* CButton::create(const char *normalImage, const char *selectedImage, CCObject* target, SEL_MenuHandler selector)
{
    return CButton::create(normalImage, selectedImage, NULL, target, selector);
}

///静态构造函数，重载 creates a menu item with a normal and selected image with target/selector 
CButton* CButton::create(const char *normalImage, const char *selectedImage, const char *disabledImage, CCObject* target, SEL_MenuHandler selector)
{
    CButton *pRet = new CButton();
    if (pRet && pRet->initWithNormalImage(normalImage, selectedImage, disabledImage, target, selector))
    {
        pRet->autorelease();
        return pRet;
    }
    CC_SAFE_DELETE(pRet);
    return NULL;
}

///静态构造函数，重载 creates a menu item with a normal,selected  and disabled image with target/selector
CButton* CButton::create(const char *normalImage, const char *selectedImage, const char *disabledImage)
{
    CButton *pRet = new CButton();
    if (pRet && pRet->initWithNormalImage(normalImage, selectedImage, disabledImage, NULL, NULL))
    {
        pRet->autorelease();
        return pRet;
    }
    CC_SAFE_DELETE(pRet);
    return NULL;
}

///设置按钮文字
void CButton::setString(const char* pString)
{
	this->createLabel();
	this->m_pLabel->setString(pString);
}

///设置按钮文字字体
void CButton::setFontName(const char* pFontName)
{
	this->createLabel();
	this->m_pLabel->setFontName(pFontName);
}

///设置按钮文字字号
void CButton::setFontSize(int nFontSize)
{
	this->createLabel();
	this->m_pLabel->setFontSize(nFontSize);
}

///设置按钮文字颜色
void CButton::setFontColor(const ccColor3B &color)
{
	this->createLabel();
	this->m_pLabel->setColor(color);
}

///如果需要按钮有文字，需要创建Label
void CButton::createLabel()
{
	if(!this->m_pLabel)
	{
		this->m_pLabel = CCLabelTTF::create();
		this->m_pLabel->setAnchorPoint(ccp(0.5,0.5));
		CCSize oSize = this->getContentSize();
		this->m_pLabel->setPosition(ccp(oSize.width/2, oSize.height/2));
		this->m_pLabel->setFontName(BUTTON_FONTNAME);
		this->m_pLabel->setFontSize(BUTTON_FONTSIZE);
		this->m_pLabel->setColor(BUTTON_FONTCOLOR);
		this->addChild(m_pLabel);
	}
}