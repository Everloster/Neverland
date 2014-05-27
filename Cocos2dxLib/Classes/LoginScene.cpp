#include "LoginScene.h"
#include "GameNetBean.h"
#include "ResourceInc.h"
#include "OppoServer.h"
#include "PokerSprite.h";

//当场景创建时调用
void CLoginScene::onCreate() 
{
	CWidgetLayout *bgLayer = new CWidgetLayout();

	/*CCMenuItemImage *bg = CCMenuItemImage::create(s_pLoginSceneAll,"",this,NULL);
	bg->setPosition(getCenterPoint());
	bg->setEnabled(true);
	bgLayer->addChild(bg);*/

	CCSize ccSize = CCDirector::sharedDirector()->getWinSize();

	CCScale9Sprite* pNickNameTextFieldBg = CCScale9Sprite::create("textfield.png");

	m_pNickNameEdit = CCEditBox::create(CCSizeMake(400,70), pNickNameTextFieldBg);
	m_pNickNameEdit->setPosition(ccp(530,300));
	m_pNickNameEdit->setFontName("Helvetica");
	m_pNickNameEdit->setFontSize(30);
	m_pNickNameEdit->setFontColor(ccBLACK);
	m_pNickNameEdit->setPlaceHolder("nikename:");
	m_pNickNameEdit->setPlaceholderFontColor(ccRED);
	m_pNickNameEdit->setMaxLength(12);
	m_pNickNameEdit->setReturnType(kKeyboardReturnTypeDone);

	CLabel *pLabeNickName = CLabel::create("NickName", 30);
	pLabeNickName->setPosition(ccp(220, 300));

	m_pExitButton = CButton::create("CloseNormal.png", "CloseSelected.png", this, menu_selector(CLoginScene::onClick));
	m_pExitButton->setAnchorPoint(ccp(1,0));
	m_pExitButton->setPosition(ccp(ccSize.width, 0));

	m_pLoginButton = CButton::create("btn_go_0.png", "btn_go_1.png", this, menu_selector(CLoginScene::onClick));
	m_pLoginButton->setPosition(620, 150);
	m_pLoginButton->setString("Login");
	m_pLoginButton->setFontSize(30);

	bgLayer->addChild(pLabeNickName);
	bgLayer->addChild(m_pExitButton);
	bgLayer->addChild(m_pLoginButton);
	bgLayer->addChild(m_pNickNameEdit);
	this->addChild(bgLayer);
}

void CLoginScene::onSettingsClick(CCObject *pSender)
{
	CCLOG("on click");
	CSceneManager::sharedSceneManager()->pushScene(GETSCENE(CMainScene));
}

//当场景接收到消息时调用
void CLoginScene::onMessage(int nMsg, ARRAY wParam, PARAM lParam)
{
	switch(nMsg)
	{
	//连接成功
	case MSG_UI_CONNECTED:
		this->login();
		break;
	//连接中断
	case MSG_UI_DISCONNECTED:
		CSceneManager::sharedSceneManager()->openPopupBox(GETBOX(CMessageBox), (void*)"disconnected!");
		break;
	//连错错误
	case MSG_UI_CONNECTERROR:
		CSceneManager::sharedSceneManager()->openPopupBox(GETBOX(CMessageBox), (void*)"connect error!");
		break;
	//连接超时
	case MSG_UI_CONNECTTIMEOUT:
		CSceneManager::sharedSceneManager()->openPopupBox(GETBOX(CMessageBox), (void*)"connect timeout!");
		break;
	//登陆成功
	case MSG_SC_INITCLIENT:
		CSceneManager::sharedSceneManager()->closePopupBox(GETBOX(CMessageBox));
		CSceneManager::sharedSceneManager()->pushScene(CCTransitionFlipX::create(0.5, GETSCENE(CSimpleGameScene)));
		break;
	}
}

void CLoginScene::onClick(CCObject *pSender)
{
	//退出按钮
	if(pSender == m_pExitButton)
	{
		//CSceneManager::sharedSceneManager()->exit();
		this->testUrl();
	}
	//登陆按钮
	else if(pSender == m_pLoginButton)
	{
		if(CGameNetBean::sharedNetBean()->isConnected())
		{
			this->login();
		}
		else
		{
			CGameNetBean::sharedNetBean()->connect();
		}
	}
}

void CLoginScene::login()
{
	const char* pNickName = m_pNickNameEdit->getText();
	if(strlen(pNickName) > 0) {
		CGameNetBean::sharedNetBean()->cs_login(pNickName);
	} else {
		CSceneManager::sharedSceneManager()->openPopupBox(GETBOX(CMessageBox), (void*)"type your name!");
	}
}

void CLoginScene::testUrl()
{
	CSceneManager::sharedSceneManager()->openPopupBox(GETBOX(CLoadingBox), (void*)(""));
	OppoServer* pServer = OppoServer::shareOppoServer();
	CCHttpRequest* request = CHttpServer::getGetRequestByUrl("http://www.bsdfsaidu.com");
	request->setResponseCallback(this, callfuncND_selector(CLoginScene::urlCallback));
	pServer->send(request);
	request->release();
}

void CLoginScene::urlCallback(CCNode *sender ,void *data)
{
	CSceneManager::sharedSceneManager()->closePopupBox(GETBOX(CLoadingBox));
	CCHttpResponse* response = (CCHttpResponse*) data;
	if (!response) 
	{
		CSceneManager::sharedSceneManager()->openPopupBox(GETBOX(CMessageBox), (void*)"failure");
		return;
	}
	int statusCode = response->getResponseCode();
	if (response->isSucceed()) 
	{
		
		CSceneManager::sharedSceneManager()->openPopupBox(GETBOX(CMessageBox), (void*)"success");
	} else 
	{
		CSceneManager::sharedSceneManager()->openPopupBox(GETBOX(CMessageBox), (void*)"failure");
	}
}