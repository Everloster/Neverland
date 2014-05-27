/******************************************************************************
文件名: togglebutton.cpp
编写者: csdn_viva@foxmail.com ccwidget1.0
编写日期: 2013-6-24
功能描述: 开关按钮的实现
******************************************************************************/
#include "widget.h"
#include "togglebutton.h"

#define TOGGLE_BUTTON_NORMALTAG   0x1
#define TOGGLE_BUTTON_CHECKEDTAG  0x2
#define TOGGLE_BUTTON_SELECTEDTAG 0x3
#define TOGGLE_BUTTON_DISABLETAG  0x4
#define SAFE_SET_VISABLE(__node__ ,__visible__) if(__node__) __node__->setVisible(__visible__)

///获取正常状态下的图片
CCNode* CToggleButton::getNormalImage()
{
    return m_pNormalImage;
}

///设置正常状态下的图片
void CToggleButton::setNormalImage(CCNode* pImage)
{
    if (pImage != m_pNormalImage) {
        if (pImage) {
            addChild(pImage, 0, TOGGLE_BUTTON_NORMALTAG);
            pImage->setAnchorPoint(ccp(0, 0));
        }
        if (m_pNormalImage) {
            removeChild(m_pNormalImage, true);
        }
        m_pNormalImage = pImage;
        this->setContentSize(m_pNormalImage->getContentSize());
        this->updateImagesVisibility();
    }
}

///获取开关被按下后的状态
CCNode* CToggleButton::getCheckedImage()
{
    return m_pCheckedImage;
}

///设置开关被按下后的状态
void CToggleButton::setCheckedImage(CCNode* pImage)
{
    if (pImage != m_pNormalImage) {
        if (pImage) {
            addChild(pImage, 0, TOGGLE_BUTTON_CHECKEDTAG);
            pImage->setAnchorPoint(ccp(0, 0));
        }
        if (m_pCheckedImage) {
            removeChild(m_pCheckedImage, true);
        }
        m_pCheckedImage = pImage;
        this->updateImagesVisibility();
    }
}

///获取开关正在被按的时候的状态
CCNode* CToggleButton::getSelectedImage()
{
    return m_pSelectedImage;
}

///设置开关正在被按的时候的状态
void CToggleButton::setSelectedImage(CCNode* pImage)
{
    if (pImage != m_pNormalImage){
        if (pImage){
            addChild(pImage, 0, TOGGLE_BUTTON_SELECTEDTAG);
            pImage->setAnchorPoint(ccp(0, 0));
        }
        if (m_pSelectedImage){
            removeChild(m_pSelectedImage, true);
        }
        m_pSelectedImage = pImage;
        this->updateImagesVisibility();
    }
}

///获取禁用时的状态
CCNode* CToggleButton::getDisabledImage()
{
    return m_pDisabledImage;
}

///设置禁用时的状态
void CToggleButton::setDisabledImage(CCNode* pImage)
{
    if (pImage != m_pNormalImage) {
        if (pImage) {
            addChild(pImage, 0, TOGGLE_BUTTON_DISABLETAG);
            pImage->setAnchorPoint(ccp(0, 0));
        }
        if (m_pDisabledImage) {
            removeChild(m_pDisabledImage, true);
        }
        m_pDisabledImage = pImage;
        this->updateImagesVisibility();
    }
}

///静态构造函数
CToggleButton* CToggleButton::create()
{
	CToggleButton *pRet = new CToggleButton();
    if (pRet && pRet->init())
    {
        pRet->autorelease();
        return pRet;
    }
    CC_SAFE_DELETE(pRet);
    return NULL;
}

///静态构造函数，重载
CToggleButton* CToggleButton::create(const char *normalImage, const char *checkedImage, const char *selectedImage/*=NULL*/, const char *disabledImage/*=NULL*/)
{
	return CToggleButton::create(normalImage, checkedImage, selectedImage, disabledImage, NULL, NULL);
}

///静态构造函数，重载
CToggleButton* CToggleButton::create(const char *normalImage, const char *checkedImage, CCObject* target, SEL_MenuHandler selector)
{
	return CToggleButton::create(normalImage, checkedImage, NULL, NULL, target, selector);
}

///静态构造函数，重载
CToggleButton* CToggleButton::create(const char *normalImage, const char *checkedImage, const char *selectedImage, const char *disabledImage, CCObject* target, SEL_MenuHandler selector)
{
	CToggleButton *pRet = new CToggleButton();
	if(pRet && pRet->initWithFile(normalImage, checkedImage, selectedImage, disabledImage, target, selector))
	{
		pRet->autorelease();
		return pRet;
	}
	CC_SAFE_DELETE(pRet);
	return NULL;
}

///初始化时调用
bool CToggleButton::init()
{
	return initWithFile(NULL, NULL, NULL, NULL, NULL, NULL);
}

///初始化时调用
bool CToggleButton::initWithFile(const char *normalImage, const char *checkedImage, const char *selectedImage, const char *disabledImage, CCObject* target, SEL_MenuHandler selector)
{
	if(!normalImage && !checkedImage){
		return false;
	}
	CCSprite *pNormalSprite   = CCSprite::create(normalImage);
	CCSprite *pCheckedSprite  = CCSprite::create(checkedImage);
	CCSprite *pSelectedSprite = NULL;
	CCSprite *pDisableSprite  = NULL;

	if(selectedImage) {
		pSelectedSprite = CCSprite::create(selectedImage);
	}
	if(disabledImage){
		pDisableSprite = CCSprite::create(disabledImage);
	}

	this->setNormalImage(pNormalSprite);
	this->setCheckedImage(pCheckedSprite);
	this->setSelectedImage(pSelectedSprite);
	this->setDisabledImage(pDisableSprite);
	
	if(m_pNormalImage)
		this->setContentSize(m_pNormalImage->getContentSize());

	SAFE_SET_VISABLE(m_pNormalImage, true);
	SAFE_SET_VISABLE(m_pCheckedImage, false);
	SAFE_SET_VISABLE(m_pSelectedImage, false);
	SAFE_SET_VISABLE(m_pDisabledImage, false);
	this->m_bIsChecked = false;
    
    setCascadeColorEnabled(true);
    setCascadeOpacityEnabled(true);

	this->initWithTarget(target, selector);

	return true;
}

///当选中时调用
void CToggleButton::selected()
{
	CCMenuItem::selected();
	//处于按下状态中
	if(m_pSelectedImage) {	
		SAFE_SET_VISABLE(m_pNormalImage, false);
		SAFE_SET_VISABLE(m_pCheckedImage, false);
		SAFE_SET_VISABLE(m_pSelectedImage, true);
		SAFE_SET_VISABLE(m_pDisabledImage, false);
	}
}

///当失去选中时调用
void CToggleButton::unselected()
{
	CCMenuItem::unselected();
	//如果抬起手指后，按照之前的选中状态恢复
	if (m_bIsChecked) {
		SAFE_SET_VISABLE(m_pNormalImage, false);
		SAFE_SET_VISABLE(m_pCheckedImage, true);
		SAFE_SET_VISABLE(m_pSelectedImage, false);
		SAFE_SET_VISABLE(m_pDisabledImage, false);
	} else {
		SAFE_SET_VISABLE(m_pNormalImage, true);
		SAFE_SET_VISABLE(m_pCheckedImage, false);
		SAFE_SET_VISABLE(m_pSelectedImage, false);
		SAFE_SET_VISABLE(m_pDisabledImage, false);
	}
}

///事件顺利执行完成，执行回调
void CToggleButton::activate()
{
	//是否可以执行事件
	bool bIsActivate = false;

	if (m_bEnabled)
    {
		//如果抬起手指后，之前的选中状态
		if (m_bIsChecked)  {
			//首先判断是否有互斥组
			if(m_pToggleGroup) {
				//如果有互斥组，就还是为选中状态，不变
				SAFE_SET_VISABLE(m_pNormalImage, false);
				SAFE_SET_VISABLE(m_pCheckedImage, true);
				SAFE_SET_VISABLE(m_pSelectedImage, false);
				SAFE_SET_VISABLE(m_pDisabledImage, false);
			} else {
				//没有互斥组 是已经选中的，那就置为不选中
				SAFE_SET_VISABLE(m_pNormalImage, true);
				SAFE_SET_VISABLE(m_pCheckedImage, false);
				SAFE_SET_VISABLE(m_pSelectedImage, false);
				SAFE_SET_VISABLE(m_pDisabledImage, false);
				bIsActivate  = true;
				m_bIsChecked = false;
			}
		} else {
			//判断是否有互斥组
			if(m_pToggleGroup) {
				//如果有互斥组，把其他按钮置为没选中
				m_pToggleGroup->setUnSelected(this);
			}
			//是没选中，就置为选中状态
			SAFE_SET_VISABLE(m_pNormalImage, false);
			SAFE_SET_VISABLE(m_pCheckedImage, true);
			SAFE_SET_VISABLE(m_pSelectedImage, false);
			SAFE_SET_VISABLE(m_pDisabledImage, false);
			bIsActivate  = true;
			m_bIsChecked = true;
		}

		if(bIsActivate)
		{
			if (m_pListener && m_pfnSelector) {
				(m_pListener->*m_pfnSelector)(this);
			}
			if (kScriptTypeNone != m_eScriptType) {
				CCScriptEngineManager::sharedManager()->getScriptEngine()->executeMenuItemEvent(this);
			}
		}
    }
}

///设置是否可用
void CToggleButton::setEnabled(bool bEnabled)
{
	if( m_bEnabled != bEnabled ) 
    {
        CCMenuItem::setEnabled(bEnabled);
		m_bIsChecked = false;
		updateImagesVisibility();
    }
}

///设置状态
void CToggleButton::setChecked(bool isChecked)
{
	if( m_bIsChecked != isChecked )
	{
		m_bIsChecked = isChecked;
		updateImagesVisibility();
	}
}

///更新状态
void CToggleButton::updateImagesVisibility()
{
	//如果不是禁用状态
	if(m_bEnabled) {
		//如果是选中状态
		if(m_bIsChecked) {	
			//判断是否有互斥组
			if(m_pToggleGroup) {
				this->m_bIsChecked = !this->m_bIsChecked;
				this->activate();
			}
			SAFE_SET_VISABLE(m_pNormalImage, false);
			SAFE_SET_VISABLE(m_pCheckedImage, true);
			SAFE_SET_VISABLE(m_pSelectedImage, false);
			SAFE_SET_VISABLE(m_pDisabledImage, false);
		} else {
			//如果不是选中状态
			SAFE_SET_VISABLE(m_pNormalImage, true);
			SAFE_SET_VISABLE(m_pCheckedImage, false);
			SAFE_SET_VISABLE(m_pSelectedImage, false);
			SAFE_SET_VISABLE(m_pDisabledImage, false);
		}
	} else {
		//如果是禁用状态
		if(m_pDisabledImage) {
			SAFE_SET_VISABLE(m_pNormalImage, false);
			SAFE_SET_VISABLE(m_pCheckedImage, false);
			SAFE_SET_VISABLE(m_pSelectedImage, false);
			SAFE_SET_VISABLE(m_pDisabledImage, true);
		} else {
			SAFE_SET_VISABLE(m_pNormalImage, true);
			SAFE_SET_VISABLE(m_pCheckedImage, false);
			SAFE_SET_VISABLE(m_pSelectedImage, false);
			SAFE_SET_VISABLE(m_pDisabledImage, false);
		}
	}
}


///构造方法，初始化指针
CToggleGroup::CToggleGroup(): 
m_pToggleChildArrs(NULL)
{

}

///析构方法，释放Array
CToggleGroup::~CToggleGroup()
{
	CC_SAFE_RELEASE_NULL(m_pToggleChildArrs);
}

///静态构造函数
CToggleGroup* CToggleGroup::create()
{
	CToggleGroup *pRet = new CToggleGroup();
	if(pRet && pRet->init())
	{
		pRet->setPosition(CCPointZero);
		pRet->autorelease();
		return pRet;
	}
	CC_SAFE_DELETE(pRet);
	return NULL;
}

///初始化时调用
bool CToggleGroup::init()
{
	CWidgetLayout::init();
	m_pToggleChildArrs = CCArray::create();
	m_pToggleChildArrs->retain();
	return true;
}

///添加互斥按钮
void CToggleGroup::addChild(CToggleButton *pChild)
{
	if(!m_pToggleChildArrs->containsObject(pChild))
	{
		m_pToggleChildArrs->addObject(pChild);
		pChild->addToggleGroup(this);
		CWidgetLayout::addChild(pChild);
	}
}

///重载removeChild
void CToggleGroup::removeChild(CToggleButton* pChild)
{
	this->removeChild(pChild, true);
}

///重载removeChild
void CToggleGroup::removeChild(CToggleButton* pChild, bool bCleanup)
{
	m_pToggleChildArrs->removeObject(pChild, true);
	CWidgetLayout::removeChild(pChild, bCleanup);
}

///重载removeChild
void CToggleGroup::removeChildByTag(int nTag)
{
	this->removeChildByTag(nTag, true);
}

///重载removeChild
void CToggleGroup::removeChildByTag(int nTag, bool bCleanup)
{
	CCObject *pObject = NULL;
	CCARRAY_FOREACH(m_pToggleChildArrs, pObject)
	{
		CToggleButton *pButton = (CToggleButton*) pObject;
		if(pButton->getTag() == nTag)
		{
			m_pToggleChildArrs->removeObject(pButton);
			CWidgetLayout::removeChild(pButton, bCleanup);
			break;
		}
	}
}

///重载removeChild
void CToggleGroup::removeAllChildren()
{
	m_pToggleChildArrs->removeAllObjects();
	CWidgetLayout::removeAllChildren();
}

///重载removeChild
void CToggleGroup::removeAllChildrenWithCleanup(bool bCleanup)
{
	m_pToggleChildArrs->removeAllObjects();
	CWidgetLayout::removeAllChildrenWithCleanup(bCleanup);
}

///把互斥锁里其他的按钮置为不选中状态
void CToggleGroup::setUnSelected(CToggleButton *pChild)
{
	if(m_pToggleChildArrs)
	{
		CCObject *pObject = NULL;
		CCARRAY_FOREACH(m_pToggleChildArrs, pObject)
		{
			if(pObject != pChild)
			{
				CToggleButton *pToggleButton = (CToggleButton*) pObject;
				if(pToggleButton->isEnabled())
				{
					pToggleButton->setChecked(false);
				}
			}
		}
	}
}