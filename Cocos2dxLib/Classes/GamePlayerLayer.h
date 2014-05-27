#ifndef __GAMEPLAYER_LAYER_H__
#define __GAMEPLAYER_LAYER_H__

#include "ccbase/base.h"
#include "Player.h"

class CGamePlayerLayer : public CCLayer
{
public:
	CREATE_FUNC(CGamePlayerLayer);

public:
	//当场景创建时调用
	virtual bool init();
	//当场景显示时调用
	virtual void onEnter();
	//当场景隐藏时调用
	virtual void onExit();

public:
	//注册事件
	virtual void registerWithTouchDispatcher();
	//按下
	virtual bool ccTouchBegan(CCTouch *pTouch, CCEvent *pEvent);
	//抬起
    virtual void ccTouchEnded(CCTouch *pTouch, CCEvent *pEvent);

private:
	//玩家对象
	CPlayerSprite *pPlayerSprite;
	//其他玩家对象组
	map<int, CPlayerSprite*> m_mOtherPlayerObj;

public:
	//生产已经存在的玩家
	void createExistsPlayer();
	//生产一个其他玩家
	void createOtherPlayer(int nID);
	//移除一个玩家
	void removeOtherPlayer(int nID);
	//其他玩家走路
	void otherPlayerMove(int nID, int nX, int nY);

};

#endif //__GAMEPLAYER_LAYER_H__