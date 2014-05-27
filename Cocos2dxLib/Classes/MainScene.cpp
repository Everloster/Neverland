#include "MainScene.h"
#include "GameNetBean.h"
#include "ccwidget/widget.h"
#include "PokerSprite.h";
#include "ResourceInc.h"
//当场景创建时调用
void CMainScene::onCreate()
{
	CCSize ccSize = CCDirector::sharedDirector()->getWinSize();
	CCPoint centerPoint = ccp(ccSize.width/2, ccSize.height/2);
	// add bg and table
	CCLayer *background = CCLayer::create();
	CCSprite *bg = CCSprite::create(s_pGameRoomSceneBg);
	CCSprite *table = CCSprite::create(s_pGameRoomSceneTable);
	bg->setPosition(centerPoint);
	table->setPosition(ccp(640,280));
	background->addChild(bg);
	background->addChild(table);
	this->addChild(background);
	// add ui layer
	UILayer *uiLayer = UILayer::create();
	Layout *uiLayout = dynamic_cast<Layout*>(CCUIHELPER->createWidgetFromJsonFile("DDZ_UI_GAME.json"));
	uiLayer->addWidget(uiLayout);
	this->addChild(uiLayer);

	// add pokers
	cardShowList *m_cardList = new cardShowList();
	PokerSprite **ps;int i=0;
	for (ps = (m_cardList->cardUIList);ps <(m_cardList->cardUIList)+10; ps++)
	{
		*ps=PokerSprite::create(3,13,m_cardList);//,m_cardList));
		this->addChild(*ps,9);
		(*ps)->setPosition(ccp(300+m_cardList->cardGap*i,100));
		i++;
	}
	m_cardList->length=10;

	// add back buttton
	pBackFont = CCMenuItemFont::create("back", this, menu_selector(CMainScene::onClick));
	pBackFont->setAnchorPoint(ccp(1,0));
	pBackFont->setPosition(ccp(CCDirector::sharedDirector()->getWinSize().width, 0));
	CWidgetLayout *pLayout = CWidgetLayout::create();
	pLayout->addChild(pBackFont);
	this->addChild(pLayout);
}

void CMainScene::onClick(CCObject *pSender)
{
	//返回按钮，断开网络，回到登陆
	CGameNetBean::sharedNetBean()->close();
	CSceneManager::sharedSceneManager()->popScene();
}

//当场景接收到消息时调用
void CMainScene::onMessage(int nMsg, ARRAY wParam, PARAM lParam)
{
	switch(nMsg)
	{
	case MSG_SC_OTHERPLAYER_RUN:
		pGameLayer->otherPlayerMove(
			wParam->GetInt(0),
			wParam->GetInt(1),
			wParam->GetInt(2));
		break;
	case MSG_SC_OTHERPLAYER_JOIN:
		pGameLayer->createOtherPlayer(wParam->GetInt(0));
		break;
	case MSG_SC_OTHERPLAYER_EXIT:
		pGameLayer->removeOtherPlayer(wParam->GetInt(0));
		break;
	//中断连接
	case MSG_UI_DISCONNECTED:
		CSceneManager::sharedSceneManager()->popScene();
		break;
	}
}