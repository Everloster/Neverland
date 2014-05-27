/******************************************************************************
文件名: togglebutton.h
编写者: csdn_viva@foxmail.com ccwidget1.0
编写日期: 2013-6-24
功能描述: 开关按钮的定义
******************************************************************************/
#ifndef __CCWIDGET_TOGGLE_BUTTON_H__
#define __CCWIDGET_TOGGLE_BUTTON_H__

/******************************************************************************
类    名: CToggleButton
功能描述: 开关按钮的定义，可以设置禁用，Check，正常状态，按下状态，选择状态等
******************************************************************************/
class CToggleButton : public CCMenuItem
{
	///正常状态下的图片
    CC_PROPERTY(CCNode*, m_pNormalImage, NormalImage);
	///开关被按下后的状态
    CC_PROPERTY(CCNode*, m_pCheckedImage, CheckedImage);
    ///开关正在被按的时候的状态
    CC_PROPERTY(CCNode*, m_pSelectedImage, SelectedImage);
    ///禁用时的状态
    CC_PROPERTY(CCNode*, m_pDisabledImage, DisabledImage);

public:
	///初始化数据
	CToggleButton():
	m_nIdx(0),
	m_bIsChecked(false),
	m_pNormalImage(NULL),
	m_pCheckedImage(NULL),
	m_pSelectedImage(NULL),
	m_pDisabledImage(NULL),
	m_pToggleGroup(NULL)
	{ };

public:
	///静态构造函数
    static CToggleButton* create();
	///静态构造函数，重载
    static CToggleButton* create(const char *normalImage, const char *checkedImage, const char *selectedImage = NULL, const char *disabledImage = NULL);
	///静态构造函数，重载
    static CToggleButton* create(const char *normalImage, const char *checkedImage, CCObject* target, SEL_MenuHandler selector);
	///静态构造函数，重载
    static CToggleButton* create(const char *normalImage, const char *checkedImage, const char *selectedImage, const char *disabledImage, CCObject* target, SEL_MenuHandler selector);

	///初始化时调用
	virtual bool init();
	///初始化时调用
    virtual bool initWithFile(const char *normalImage, const char *checkedImage, const char *selectedImage, const char *disabledImage, CCObject* target, SEL_MenuHandler selector);

public:
	///当选中时调用
	virtual void selected();
	///当失去选中时调用
    virtual void unselected();
	///事件顺利执行完成，执行回调
    virtual void activate();
	///设置是否可用
    virtual void setEnabled(bool bEnabled);
	///更新状态
	virtual void updateImagesVisibility();

public:
	///from parent class to impl
	virtual void setOpacityModifyRGB(bool bValue) {CC_UNUSED_PARAM(bValue);}
	///from parent class to impl
    virtual bool isOpacityModifyRGB(void) { return false;}

public:
	///舍去的方法，暂时不支持
	virtual void removeFromParent() {}
    ///舍去的方法，暂时不支持
	virtual void removeFromParentAndCleanup(bool cleanup) {}

public:
	///检测是否是按下状态
	virtual bool isChecked() { return m_bIsChecked; };
	///设置状态
	virtual void setChecked(bool isChecked);

public:
	///获得下标ID
	virtual int getIdx(){ return m_nIdx; }
	///设置下标ID
	virtual void setIdx(int idx){ m_nIdx = idx; }
	///添加一个互斥组
	virtual void addToggleGroup(CToggleGroup *pToggleGroup) { m_pToggleGroup = pToggleGroup; }

protected:
	///是否是按下状态
	bool m_bIsChecked;
	///下标ID,用于列表控件中标识
	int m_nIdx;

	///互斥组
	CToggleGroup *m_pToggleGroup;
};

/******************************************************************************
类    名: CToggleGroup
功能描述: 开关按钮的互斥组
******************************************************************************/
class CToggleGroup : public CWidgetLayout
{
public:
	///构造方法，初始化指针
	CToggleGroup();
	///析构方法，释放Array
	virtual ~CToggleGroup();

public:
	///初始化时调用
	virtual bool init();
	///添加互斥按钮
	virtual void addChild(CToggleButton *pChild);
	///把互斥锁里其他的按钮置为不选中状态
	virtual void setUnSelected(CToggleButton *pChild);
	///获得所有要相对互斥的开关按钮数组
	virtual CCArray* getToggleArray() { return m_pToggleChildArrs;}

public:
	///重载removeChild
	virtual void removeChild(CToggleButton* pChild);
    ///重载removeChild
    virtual void removeChild(CToggleButton* pChild, bool bCleanup);
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

public:
	///静态构造函数
	static CToggleGroup* create();

protected:
	///保存所有要相对互斥的开关按钮
	CCArray *m_pToggleChildArrs;
};

#endif //__CCWIDGET_TOGGLE_BUTTON_H__