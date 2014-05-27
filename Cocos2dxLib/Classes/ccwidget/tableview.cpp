/******************************************************************************
文件名: tableview.cpp
编写者: csdn_viva@foxmail.com ccwidget1.0
编写日期: 2013-6-5
功能描述: 滑动列表控件实现。
******************************************************************************/
#include "widget.h"
#include "tableview.h"

///静态构造函数，通过数据回调接口和size构建
CTableView* CTableView::create(CCTableViewDataSource *pDataSource, CCSize tSize)
{
	CTableView *table = new CTableView();
	if(table && table->initWithViewSize(tSize, NULL))
	{
		table->autorelease();
		table->setDataSource(pDataSource);
		return table;
	}
	CC_SAFE_DELETE(table);
	return NULL;
}

///重写事件分发函数，使tableview的优先级可控制
void CTableView::registerWithTouchDispatcher()
{
	CCDirector::sharedDirector()->getTouchDispatcher()->addTargetedDelegate(this,this->getTouchPriority(),true);
}

///重写当触摸按下时回调
bool CTableView::ccTouchBegan(CCTouch *pTouch, CCEvent *pEvent)
{
	if (!this->isVisible()) {
        return false;
    }

    bool touchResult = CCScrollView::ccTouchBegan(pTouch, pEvent);

    if(m_pTouches->count() == 1) {
        unsigned int      index;
        CCPoint           point;

        point = this->getContainer()->convertTouchToNodeSpace(pTouch);

        index = this->_indexFromOffset(point);
		if (index == CC_INVALID_INDEX)
		{
			m_pTouchedCell = NULL;
		}
        else
		{
			m_pTouchedCell  = this->cellAtIndex(index);
		}

        if (m_pTouchedCell) {
			if (m_pTableViewDelegate != NULL) {
				m_pTableViewDelegate->tableCellHighlight(this, m_pTouchedCell);
			}
			((CTableViewCell*)m_pTouchedCell)->selected(m_pTouchedCell->convertTouchToNodeSpace(pTouch));    
        }
    }
    else if(m_pTouchedCell) {
        if (m_pTableViewDelegate != NULL) {
            m_pTableViewDelegate->tableCellUnhighlight(this, m_pTouchedCell);
        }
		((CTableViewCell*)m_pTouchedCell)->unselected();
        m_pTouchedCell = NULL;
    }

    return touchResult;
}

///重写当移动时回调
void CTableView::ccTouchMoved(CCTouch *pTouch, CCEvent *pEvent)
{
	CCScrollView::ccTouchMoved(pTouch, pEvent);

    if (m_pTouchedCell && isTouchMoved()) {
        if(m_pTableViewDelegate != NULL) {
            m_pTableViewDelegate->tableCellUnhighlight(this, m_pTouchedCell);
        }
		((CTableViewCell*)m_pTouchedCell)->unselected();
        m_pTouchedCell = NULL;
    }
}

///重写当触摸释放时调用
void CTableView::ccTouchEnded(CCTouch *pTouch, CCEvent *pEvent)
{
	if (!this->isVisible()) {
        return;
    }

    if (m_pTouchedCell){
		CCRect bb = this->boundingBox();
		bb.origin = m_pParent->convertToWorldSpace(bb.origin);

		if (bb.containsPoint(pTouch->getLocation()))
        {
			if (m_pTableViewDelegate != NULL) {
				 m_pTableViewDelegate->tableCellUnhighlight(this, m_pTouchedCell);
			}
			//让菜单处理事件
			bool bIsActivate = ((CTableViewCell*)m_pTouchedCell)->activate();
			//如果没有菜单项处理事件，就把事件交给代理
			if (m_pTableViewDelegate != NULL && !bIsActivate) {
				m_pTableViewDelegate->tableCellTouched(this, m_pTouchedCell);
			}
        }

        m_pTouchedCell = NULL;
    }

    CCScrollView::ccTouchEnded(pTouch, pEvent);
	//自动对齐到页
	this->adjust();
}

///重写当触摸中断时调用
void CTableView::ccTouchCancelled(CCTouch *pTouch, CCEvent *pEvent)
{
	CCScrollView::ccTouchCancelled(pTouch, pEvent);

    if (m_pTouchedCell) {
        if(m_pTableViewDelegate != NULL) {	
            m_pTableViewDelegate->tableCellUnhighlight(this, m_pTouchedCell);		
        }
		((CTableViewCell*)m_pTouchedCell)->unselected();
        m_pTouchedCell = NULL;
    }
	//自动对齐到页
	this->adjust();
}

///自动对齐到页
void CTableView::adjust()
{
	//是否需要自动对齐
	if(m_bIsAutoAdjust)
	{
		//得到可视范围的宽度
		int nViewWidth = this->getViewSize().width;
		//关闭CCScrollView中的自调整
		this->unscheduleAllSelectors();
    
		int x = this->getContentOffset().x;
		int offset = (int) x % nViewWidth;
		// 调整位置
		CCPoint adjustPos;
		// 调整动画时间
		float adjustAnimDelay;
    
		if (offset < -(nViewWidth / 2)) {
			// 计算下一页位置，时间
			adjustPos = ccpSub(this->getContentOffset(), ccp(nViewWidth + offset, 0));
			adjustAnimDelay = (float) (nViewWidth + offset) / m_nAdjustInterval;
		}
		else { 
			// 计算当前页位置，时间
			adjustPos = ccpSub(this->getContentOffset(), ccp(offset, 0));
			// 这里要取绝对值，否则在第一页往左翻动的时，保证adjustAnimDelay为正数
			adjustAnimDelay = (float) abs(offset) / m_nAdjustInterval;
		}
		// 调整位置
		this->setContentOffsetInDuration(adjustPos, adjustAnimDelay);
	}
}


///构造，初始化数据
CTableViewCell::CTableViewCell()
{
	m_pTouchItem = NULL;
	m_pArrItems = CCArray::create();
	m_pArrItems->retain();
}

///析构，释放资源
CTableViewCell::~CTableViewCell()
{
	CC_SAFE_RELEASE(m_pArrItems);
}

///重载addChild
void CTableViewCell::addChild(CCNode *pChild)
{
	addChild(pChild, pChild->getZOrder(), pChild->getTag());
}

///重载addChild
void CTableViewCell::addChild(CCNode *pChild, int zOrder)
{
	addChild(pChild, zOrder, pChild->getTag());
}

///重载addChild
void CTableViewCell::addChild(CCNode *pChild, int zOrder, int nTag)
{
	CCTableViewCell::addChild(pChild, zOrder, nTag);
}

///重载addChild,添加MENU控件,用于分发事件
void CTableViewCell::addChild(CCMenuItem *pChild)
{
	addChild(pChild, pChild->getZOrder(), pChild->getTag());
}

///重载addChild,添加MENU控件,用于分发事件
void CTableViewCell::addChild(CCMenuItem* pChild, int zOrder)
{
	addChild(pChild, zOrder, pChild->getTag());
}

///重载addChild,添加MENU控件,用于分发事件
void CTableViewCell::addChild(CCMenuItem* pChild, int zOrder, int nTag)
{
	m_pArrItems->addObject(pChild);
	CCTableViewCell::addChild(pChild, zOrder, nTag);
}

///当点击CELL之后，判断是否点击到了某个菜单项，并置为选中状态
void CTableViewCell::selected(const CCPoint &touchPoint)
{
	CCObject *pObject = NULL;
	CCARRAY_FOREACH_REVERSE(m_pArrItems, pObject) 
	{
		CCMenuItem *pMenuItem = (CCMenuItem *) pObject;
		if(pMenuItem->isEnabled() && pMenuItem->isVisible())
		{
			if(pMenuItem->boundingBox().containsPoint(touchPoint))
			{
				pMenuItem->selected();
				m_pTouchItem = pMenuItem;
				break;
			}
		}
	}
}

///取消了事件，把选中的菜单项置为未选中
void CTableViewCell::unselected()
{
	if(m_pTouchItem != NULL && m_pTouchItem->isEnabled() && m_pTouchItem->isVisible())
	{
		m_pTouchItem->unselected();
		m_pTouchItem = NULL;
	}
}

///触摸执行完成，成功执行事件，调用绑定的函数
bool CTableViewCell::activate()
{
	if(m_pTouchItem != NULL && m_pTouchItem->isEnabled() && m_pTouchItem->isVisible())
	{
		m_pTouchItem->unselected();
        m_pTouchItem->activate();
		m_pTouchItem = NULL;
		return true;
	}
	return false;
}