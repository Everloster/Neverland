#include "GamePlayerLayer.h"
#include "GameNetBean.h"

//当场景创建时调用
bool CGamePlayerLayer::init()
{
	CCLayer::init();

	//玩家
	pPlayerSprite = CPlayerSprite::create(CGameModel::sharedGameModel()->m_Player.m_strName.c_str());
	pPlayerSprite->setPosition(
		ccp(CGameModel::sharedGameModel()->m_Player.m_nX,
			CGameModel::sharedGameModel()->m_Player.m_nY));

	this->addChild(pPlayerSprite);
	this->createExistsPlayer();
	return true;
}

//当场景显示时调用
void CGamePlayerLayer::onEnter()
{
	this->setTouchEnabled(true);
	CCLayer::onEnter();
}

//当场景隐藏时调用
void CGamePlayerLayer::onExit()
{
	this->setTouchEnabled(false);
	CCLayer::onExit();
}

//生产已经存在的玩家
void CGamePlayerLayer::createExistsPlayer()
{
	CGameModel *pModel = CGameModel::sharedGameModel();

	map<int, COtherPlayer>::iterator iter = pModel->m_mOtherPlayerMap.begin();
	for(; iter != pModel->m_mOtherPlayerMap.end(); iter++)
	{
		CPlayerSprite *pOtherPlayer = CPlayerSprite::create(iter->second.m_strName.c_str());
		pOtherPlayer->setPosition(ccp(iter->second.m_nX, iter->second.m_nY));
		this->m_mOtherPlayerObj[iter->second.m_nOtherPlayerID] = pOtherPlayer;
		this->addChild(pOtherPlayer);
	}
}

//生产一个其他玩家
void CGamePlayerLayer::createOtherPlayer(int nID)
{
	CGameModel *pModel = CGameModel::sharedGameModel();
	map<int, COtherPlayer>::iterator iter = pModel->m_mOtherPlayerMap.find(nID);
	if(iter != pModel->m_mOtherPlayerMap.end())
	{
		CPlayerSprite *pOtherPlayer = CPlayerSprite::create(iter->second.m_strName.c_str());
		pOtherPlayer->setPosition(ccp(iter->second.m_nX,iter->second.m_nY));
		this->m_mOtherPlayerObj[nID] = pOtherPlayer;
		this->addChild(pOtherPlayer);
	}
}

//移除一个玩家
void CGamePlayerLayer::removeOtherPlayer(int nID)
{
	map<int, CPlayerSprite*>::iterator iter = this->m_mOtherPlayerObj.find(nID);
	if(iter != this->m_mOtherPlayerObj.end())
	{
		this->removeChild(iter->second,true);
		this->m_mOtherPlayerObj.erase(iter);
	}
}

//其他玩家走路
void CGamePlayerLayer::otherPlayerMove(int nID, int nX, int nY)
{
	map<int, CPlayerSprite*>::iterator iter = this->m_mOtherPlayerObj.find(nID);
	if(iter != this->m_mOtherPlayerObj.end())
	{
		iter->second->run(nX,nY);
	}
}

//注册事件
void CGamePlayerLayer::registerWithTouchDispatcher()
{
	CCDirector::sharedDirector()->getTouchDispatcher()->addTargetedDelegate(this, this->getTouchPriority(), true);
}

//按下
bool CGamePlayerLayer::ccTouchBegan(CCTouch *pTouch, CCEvent *pEvent)
{
	return true;
}

//抬起
void CGamePlayerLayer::ccTouchEnded(CCTouch* pTouch, CCEvent* pEvent)
{
	pPlayerSprite->run(pTouch->getLocation().x, pTouch->getLocation().y);
	CGameNetBean::sharedNetBean()->cs_run(pTouch->getLocation().x, pTouch->getLocation().y);
}