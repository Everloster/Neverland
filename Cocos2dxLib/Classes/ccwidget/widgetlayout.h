/******************************************************************************
文件名: widgetlayout.h
编写者: csdn_viva@foxmail.com ccwidget1.0
编写日期: 2013-6-5
功能描述: 定义了控件的事件分发层
******************************************************************************/
#ifndef __CCWIDGET_WIDGETLAYOUT_H__
#define __CCWIDGET_WIDGETLAYOUT_H__

#include <map>
#include <vector>
using namespace std;

/******************************************************************************
类    名: CWidgetLayout
功能描述: 菜单项的父层，用于事件分发
******************************************************************************/
class CWidgetLayout : public CCLayer
{
public:
	///构造函数，初始化指针
	CWidgetLayout();
	///析构函数，释放数据
	virtual ~CWidgetLayout();

public:
	///静态构造函数
	static CWidgetLayout* create();

public:
	///当显示时调用，注册事件接收
	virtual void onEnter();
	///当退出时调用，取消注册
	virtual void onExit();
	///注册事件接收
	virtual void registerWithTouchDispatcher();
	///当触摸按下时调用
    virtual bool ccTouchBegan(CCTouch *pTouch, CCEvent *pEvent);
	///当触摸移动时调用
    virtual void ccTouchMoved(CCTouch *pTouch, CCEvent *pEvent);
	///当结束触摸时调用
    virtual void ccTouchEnded(CCTouch *pTouch, CCEvent *pEvent);
	///当触摸被中断时调用
    virtual void ccTouchCancelled(CCTouch *pTouch, CCEvent *pEvent);

public:
	///重载addChild
	virtual void addChild(CCNode *pChild);
	///重载addChild
    virtual void addChild(CCNode *pChild, int zOrder);
	///重载addChild
    virtual void addChild(CCNode *pChild, int zOrder, int nTag);
	///重载addChild,添加MENU控件,用于分发事件
	virtual void addChild(CCMenuItem *pChild);
	///addChild,添加MENUIMAGE
	virtual void addChild(CCMenuItemImage *pChild);
	///重载addChild,添加子布局
	virtual void addChild(CWidgetLayout *pChild);

public:
	///重载removeChild
	virtual void removeChild(CCNode* pChild);
    ///重载removeChild
    virtual void removeChild(CCNode* pChild, bool bCleanup);
    ///重载removeChild
    virtual void removeChildByTag(int nTag);
    ///重载removeChild
    virtual void removeChildByTag(int nTag, bool bCleanup);
    ///重载removeChild
    virtual void removeAllChildren();
    ///重载removeChild
    virtual void removeAllChildrenWithCleanup(bool bCleanup);

public:
	///舍去的方法，暂时不支持
	virtual void removeFromParent() {}
    ///舍去的方法，暂时不支持
	virtual void removeFromParentAndCleanup(bool cleanup) {}
	
protected:
	///保存子菜单项的容器
	CCArray *m_pChildArrs;
	///保存所有子布局
	CCArray *m_pLayoutArrs;
	///被选中的菜单项
	CCMenuItem *m_pTouchMenuItem;
};

#endif //__CCWIDGET_WIDGETLAYOUT_H__