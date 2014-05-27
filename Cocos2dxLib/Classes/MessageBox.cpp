#include "MessageBox.h"

///当场景创建时调用
void CMessageBox::onCreate()
{
	CCSize winSize = CCDirector::sharedDirector()->getWinSize();
	CCSprite *pBg = CCSprite::create("message_box_bg.png");

	this->setCached(false);
	this->setTouchPriority(POPUPBOX_PRIORITY);	//最先接收到事件
	this->setContentSize(pBg->getContentSize());
	this->setPosition(ccp( (winSize.width - pBg->getContentSize().width) / 2,
		(winSize.height - pBg->getContentSize().height) / 2));

	pBg->setAnchorPoint(CCPointZero);
	pBg->setPosition(CCPointZero);
	this->addChild(pBg);

	m_pMessageLabel = CLabel::create();
	m_pMessageLabel->setFontSize(30);
	m_pMessageLabel->setPosition(ccp(this->getContentSize().width / 2, this->getContentSize().height / 2));
	this->addChild(m_pMessageLabel);

}

///当弹出框被打开的时候调用
void CMessageBox::onOpen(PARAM param)
{
	if(param == NULL)
		return;

	const char *pStr = (char*) param;
	m_pMessageLabel->setString(pStr);
}

bool CMessageBox::ccTouchBegan(CCTouch *pTouch, CCEvent *pEvent)
{
	//如果点击了弹出框以外的区域，就关闭弹出框
	CCPoint ccPoint = pTouch->getLocation();
	if(!this->boundingBox().containsPoint(ccPoint))
	{
		this->closeSelf();
		return true;
	}
	return true;
}