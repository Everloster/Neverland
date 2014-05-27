/******************************************************************************
文件名: widgetlayout.h
编写者: csdn_viva@foxmail.com ccwidget1.0
编写日期: 2013-6-24
功能描述: 定义了控件的事件分发层
******************************************************************************/
#include "widget.h"
#include "widgetlayout.h"

///构造函数，初始化指针
CWidgetLayout::CWidgetLayout()
{
	m_pTouchMenuItem = NULL;
	m_pChildArrs = CCArray::create();
	m_pChildArrs->retain();
	m_pLayoutArrs = CCArray::create();
	m_pLayoutArrs->retain();
}

///析构函数，释放数据
CWidgetLayout::~CWidgetLayout()
{
	CC_SAFE_RELEASE_NULL(m_pChildArrs);
	CC_SAFE_RELEASE_NULL(m_pLayoutArrs);
}

///静态构造函数
CWidgetLayout* CWidgetLayout::create()
{
	CWidgetLayout *pRet = new CWidgetLayout();
	if(pRet && pRet->init())
	{
		pRet->setPosition(CCPointZero);
		pRet->autorelease();
		return pRet;
	}
	CC_SAFE_DELETE(pRet);
	return NULL;
}

///重载addChild
void CWidgetLayout::addChild(CCNode *pChild)
{
	addChild(pChild, pChild->getZOrder(), pChild->getTag());
}

///重载addChild
void CWidgetLayout::addChild(CCNode *pChild, int zOrder)
{
	addChild(pChild, zOrder, pChild->getTag());
}

///重载addChild
void CWidgetLayout::addChild(CCNode *pChild, int zOrder, int nTag)
{
	CCLayer::addChild(pChild, zOrder, nTag);
}

///重载addChild,添加子布局
void CWidgetLayout::addChild(CWidgetLayout *pChild)
{
	if(!m_pLayoutArrs->containsObject(pChild))
	{
		m_pLayoutArrs->addObject(pChild);
		CCLayer::addChild(pChild);
	}
}

///重载addChild,添加MENU控件,用于分发事件
void CWidgetLayout::addChild(CCMenuItem *pChild)
{
	if(!m_pChildArrs->containsObject(pChild))
	{
		m_pChildArrs->addObject(pChild);
		CCLayer::addChild(pChild, pChild->getZOrder(), pChild->getTag());
	}
}

///重载addChild,添加MENU控件,用于分发事件
void CWidgetLayout::addChild(CCMenuItemImage *pChild)
{
	if(!m_pChildArrs->containsObject(pChild))
	{
		m_pChildArrs->addObject(pChild);
		CCLayer::addChild(pChild, pChild->getZOrder(), pChild->getTag());
	}
}

///重载removeChild
void CWidgetLayout::removeChild(CCNode* pChild)
{
	removeChild(pChild, true);
}

///重载removeChild
void CWidgetLayout::removeChild(CCNode* pChild, bool bCleanup)
{
	if(m_pChildArrs->containsObject(pChild))
	{
		m_pChildArrs->removeObject(pChild, true);
		CCLayer::removeChild(pChild, bCleanup);
		return;
	}
	if(m_pLayoutArrs->containsObject(pChild))
	{
		m_pLayoutArrs->removeObject(pChild, true);
		CCLayer::removeChild(pChild, bCleanup);
		return;
	}
	CCLayer::removeChild(pChild, bCleanup);
}

///重载removeChild
void CWidgetLayout::removeChildByTag(int nTag)
{
	removeChildByTag(nTag, true);
}

///重载removeChild
void CWidgetLayout::removeChildByTag(int nTag, bool bCleanup)
{
	CCAssert( nTag != kCCNodeTagInvalid, "Invalid tag");
	CCObject *pObject = NULL;
	CCARRAY_FOREACH(m_pChildArrs, pObject)
	{
		CCNode *pNode = (CCNode*) pObject;
		if(pNode->getTag() == nTag)
		{
			m_pChildArrs->removeObject(pNode, true);
			CCLayer::removeChild(pNode, bCleanup);
			return;
		}
	}
	CCARRAY_FOREACH(m_pLayoutArrs, pObject)
	{
		CCNode *pNode = (CCNode*) pObject;
		if(pNode->getTag() == nTag)
		{
			m_pLayoutArrs->removeObject(pNode, true);
			CCLayer::removeChild(pNode, bCleanup);
			return;
		}
	}
	CCLayer::removeChildByTag(nTag, bCleanup);
}

///重载removeChild
void CWidgetLayout::removeAllChildren()
{
	m_pChildArrs->removeAllObjects();
	m_pLayoutArrs->removeAllObjects();
	CCLayer::removeAllChildren();
}

///重载removeChild
void CWidgetLayout::removeAllChildrenWithCleanup(bool bCleanup)
{
	m_pChildArrs->removeAllObjects();
	m_pLayoutArrs->removeAllObjects();
	CCLayer::removeAllChildrenWithCleanup(bCleanup);
}

///当显示时调用，注册事件接收
void CWidgetLayout::onEnter()
{
	CCLayer::setTouchEnabled(true);
	CCLayer::onEnter();
}

///当退出时调用，取消注册
void CWidgetLayout::onExit()
{
	CCLayer::setTouchEnabled(false);
	CCLayer::onExit();
}

///注册事件接收
void CWidgetLayout::registerWithTouchDispatcher()
{
	CCDirector::sharedDirector()->getTouchDispatcher()->addTargetedDelegate(this, this->getTouchPriority(), true);
}

///当触摸按下时调用
bool CWidgetLayout::ccTouchBegan(CCTouch *pTouch, CCEvent *pEvent)
{
	CCPoint touchPoint = this->convertTouchToNodeSpace(pTouch);
	CCObject *pObject = NULL;
	CCARRAY_FOREACH_REVERSE(m_pChildArrs, pObject)
	{
		CCMenuItem *pMenuItem = (CCMenuItem*) pObject;
		if(pMenuItem->isVisible())
		{
			if(pMenuItem->boundingBox().containsPoint(touchPoint))
			{
				if(pMenuItem->isEnabled())
				{
					m_pTouchMenuItem = pMenuItem;
					m_pTouchMenuItem->selected();
					return true;
				}
			}
		}
	}
	return false;
}

///当触摸移动时调用
void CWidgetLayout::ccTouchMoved(CCTouch *pTouch, CCEvent *pEvent)
{
	CCPoint touchPoint = this->convertTouchToNodeSpace(pTouch);
	if(m_pTouchMenuItem && m_pTouchMenuItem->isVisible()) 
	{
		if(!m_pTouchMenuItem->boundingBox().containsPoint(touchPoint)) 
		{
			if(m_pTouchMenuItem->isSelected()) 
			{
				m_pTouchMenuItem->unselected();
			}
		} 
		else 
		{
			if(!m_pTouchMenuItem->isSelected()) 
			{
				m_pTouchMenuItem->selected();
			}
		}
	}
}

///当结束触摸时调用
void CWidgetLayout::ccTouchEnded(CCTouch *pTouch, CCEvent *pEvent)
{
	CCPoint touchPoint = this->convertTouchToNodeSpace(pTouch);
	if(m_pTouchMenuItem && m_pTouchMenuItem->isVisible())
	{
		if(m_pTouchMenuItem->boundingBox().containsPoint(touchPoint))
		{
			m_pTouchMenuItem->unselected();
			m_pTouchMenuItem->activate();
			m_pTouchMenuItem = NULL;
		}
	}
}

///当触摸被中断时调用
void CWidgetLayout::ccTouchCancelled(CCTouch *pTouch, CCEvent *pEvent)
{
	if(m_pTouchMenuItem && m_pTouchMenuItem->isVisible())
	{
		m_pTouchMenuItem->unselected();
		m_pTouchMenuItem = NULL;
	}
}