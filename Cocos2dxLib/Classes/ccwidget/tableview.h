/******************************************************************************
文件名: tableview.h
编写者: csdn_viva@foxmail.com ccwidget1.0
编写日期: 2013-6-5
功能描述: 滑动列表控件定义。
******************************************************************************/
#ifndef __CCWIDGET_TABLEVIEW_H__
#define __CCWIDGET_TABLEVIEW_H__

/******************************************************************************
类    名: CTableView
功能描述: 滑动列表控件类，继承与CCTableView实现其他附加的功能
******************************************************************************/
class CTableView : public CCTableView
{
public:
	///构造方法，初始化数据
	CTableView() : m_bIsAutoAdjust(false), m_nAdjustInterval(1000) { };

public:
	///静态构造函数，通过数据回调接口和size构建
	static CTableView* create(CCTableViewDataSource *pDataSource, CCSize tSize);

public:
	///设置触摸释放后 是否自动对齐到CELL
	void setAutoAdjust(bool bIsAutoAdjust){ m_bIsAutoAdjust = bIsAutoAdjust; }
	///设置对齐时的动画间隔
	void setAdjustInterval(int nAdjustInterval){ m_nAdjustInterval = nAdjustInterval; }
	///自动对齐到页
	void adjust();

public:
	///重写事件分发函数，使tableview的优先级可控制
	virtual void registerWithTouchDispatcher();

	///重写当触摸按下时回调
	virtual bool ccTouchBegan(CCTouch *pTouch, CCEvent *pEvent);
	///重写当移动时回调
    virtual void ccTouchMoved(CCTouch *pTouch, CCEvent *pEvent);
	///重写当触摸释放时调用
    virtual void ccTouchEnded(CCTouch *pTouch, CCEvent *pEvent);
	///重写当触摸中断时调用
    virtual void ccTouchCancelled(CCTouch *pTouch, CCEvent *pEvent);

protected:
	///触摸释放后 是否自动对齐到CELL
	bool m_bIsAutoAdjust;
	///自动对齐CELL时的动画间隔
	int  m_nAdjustInterval;
};



/******************************************************************************
类    名: CTableViewCell
功能描述: 滑动列表的单元格类，拓展功能
******************************************************************************/
class CTableViewCell : public CCTableViewCell
{
public:
	///构造，初始化数据
	CTableViewCell();
	///析构，释放资源
	virtual ~CTableViewCell();

public:
	///重载addChild
	virtual void addChild(CCNode *pChild);
	///重载addChild
    virtual void addChild(CCNode *pChild, int zOrder);
	///重载addChild
    virtual void addChild(CCNode *pChild, int zOrder, int nTag);
	///重载addChild,添加MENU控件,用于分发事件
	virtual void addChild(CCMenuItem *pChild);
	///重载addChild,添加MENU控件,用于分发事件
    virtual void addChild(CCMenuItem *pChild, int zOrder);
	///重载addChild,添加MENU控件,用于分发事件
    virtual void addChild(CCMenuItem *pChild, int zOrder, int nTag);

public:
	///当点击CELL之后，判断是否点击到了某个菜单项，并置为选中状态
	virtual void selected(const CCPoint &touchPoint);
	///取消了事件，把选中的菜单项置为未选中
	virtual void unselected();
	///触摸执行完成，成功执行事件，调用绑定的函数
	virtual bool activate();

protected:
	///菜单项的集合
	CCArray *m_pArrItems;
	///处于正在点击中的菜单项
	CCMenuItem *m_pTouchItem;

};

#endif // __CCWIDGET_TABLEVIEW_H__