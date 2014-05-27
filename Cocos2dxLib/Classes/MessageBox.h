#ifndef __MESSAGE_BOX_H__
#define __MESSAGE_BOX_H__

#include "ccbase/base.h"
#include "ccwidget/widget.h"

class CMessageBox : public CPopupBox
{
public:
	DEFINE_POPUPBOX_CLASS(CMessageBox);

public:
	///当场景创建时调用
	virtual void onCreate();
	///当场景被打开的时候调用
	virtual void onOpen(PARAM param);

	virtual bool ccTouchBegan(CCTouch *pTouch, CCEvent *pEvent);

private:
	CLabel *m_pMessageLabel;

};

#endif //__MESSAGE_BOX_H__