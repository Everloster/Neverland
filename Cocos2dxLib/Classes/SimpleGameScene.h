#ifndef __SIMPLEGAMESCENE_H__
#define __SIMPLEGAMESCENE_H__

#include "ccbase/base.h"
#include "GamePlayerLayer.h"

class CSimpleGameScene : public CScene
{
public:
	DEFINE_SCENE_CLASS(CSimpleGameScene);

public:
	//当场景创建时调用
	virtual void onCreate();
	//当场景接收到消息时调用
	virtual void onMessage(int nMsg, ARRAY wParam, PARAM lParam);

	void onClick(CCObject *pSender);

private:
	CGamePlayerLayer *pGameLayer;

	CCMenuItemFont *pBackFont;
};

#endif //__MAIN_SCENE_H__