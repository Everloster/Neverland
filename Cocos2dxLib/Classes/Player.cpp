#include "Player.h"

CPlayerSprite* CPlayerSprite::create(const char *pName)
{
	CPlayerSprite *pobSprite = new CPlayerSprite();
    if (pobSprite && pobSprite->init("player.png", pName))
    {
        pobSprite->autorelease();
        return pobSprite;
    }
    CC_SAFE_DELETE(pobSprite);
    return NULL;
}

bool CPlayerSprite::init(const char* pszFileName, const char* pName)
{
	if(this->initWithFile(pszFileName))
	{
		pNameLabel = CCLabelTTF::create(pName, "黑体", 30);
		pNameLabel->setPosition(ccp(this->getContentSize().width / 2, this->getContentSize().height + 20));
		this->addChild(pNameLabel);
		return true;
	}
	return false;
}

void CPlayerSprite::run(int nX, int nY)
{
	this->stopAllActions();
	CCMoveTo *pMove = CCMoveTo::create(1,ccp(nX,nY));
	this->runAction(pMove);
}