#include "GameNetBean.h"

CGameNetBean::~CGameNetBean()
{
	CSceneManager::sharedSceneManager()->removeLooper(this);
}

///当创建成功 开始连接时调用
void CGameNetBean::onCreate()
{
	CSceneManager::sharedSceneManager()->addLooper(this);
}

///当断开连接时调用
void CGameNetBean::onDisconnected()
{
	CSceneManager::sharedSceneManager()->PostMessageA(MSG_UI_DISCONNECTED, NULL, NULL);
}

///当连接错误时调用
void CGameNetBean::onConnectError()
{
	CSceneManager::sharedSceneManager()->PostMessageA(MSG_UI_CONNECTERROR, NULL, NULL);
}

///当连接超时时调用
void CGameNetBean::onConnectTimeout()
{
	CSceneManager::sharedSceneManager()->PostMessageA(MSG_UI_CONNECTTIMEOUT, NULL, NULL);
}

///当连接成功时调用
void CGameNetBean::onConnected()
{
	CSceneManager::sharedSceneManager()->PostMessageA(MSG_UI_CONNECTED, NULL, NULL);
}

///当读到消息时调用
void CGameNetBean::onMessage(STREAM stream)
{
	switch(stream.readInt())
	{
	case MSG_SC_ERROR:
		sc_error(stream);
		break;
	case MSG_SC_INITCLIENT:
		sc_initclient(stream);
		break;
	case MSG_SC_OTHERPLAYER_RUN:
		sc_otherplayer_run(stream);
		break;
	case MSG_SC_OTHERPLAYER_JOIN:
		sc_otherplayer_join(stream);
		break;
	case MSG_SC_OTHERPLAYER_EXIT:
		sc_otherplayer_exit(stream);
		break;
	default:
		break;
	}
}

void CGameNetBean::sc_error(STREAM stream)
{
	CSceneManager::sharedSceneManager()->PostMessageA(stream.readInt(), NULL, NULL);
}

void CGameNetBean::sc_initclient(STREAM stream)
{
	//处理游戏逻辑
	CGameModel *pModel = CGameModel::sharedGameModel();
	pModel->m_Player.m_nPlayerID = stream.readInt();
	pModel->m_Player.m_strName = stream.readUTF();
	pModel->m_Player.m_nX = stream.readInt();
	pModel->m_Player.m_nY = stream.readInt();

	//加载房间内所有的玩家
	int nCount = stream.readInt();
	for(int i = 0; i < nCount; i++)
	{
		COtherPlayer otherPlayer;
		otherPlayer.m_nOtherPlayerID = stream.readInt();
		otherPlayer.m_strName = stream.readUTF();
		otherPlayer.m_nX = stream.readInt();
		otherPlayer.m_nY = stream.readInt();
		pModel->m_mOtherPlayerMap[otherPlayer.m_nOtherPlayerID] = otherPlayer;
	}
	
	//发送至UI处理
	CSceneManager::sharedSceneManager()->PostMessageA(MSG_SC_INITCLIENT, NULL, NULL);
}

void CGameNetBean::sc_otherplayer_run(STREAM stream)
{
	//处理游戏逻辑
	CGameModel *pModel = CGameModel::sharedGameModel();
	int nOtherPlayerId = stream.readInt();
	int x = stream.readInt();
	int y = stream.readInt();

	map<int,COtherPlayer>::iterator iter = pModel->m_mOtherPlayerMap.find(nOtherPlayerId);
	if(iter != pModel->m_mOtherPlayerMap.end())
	{
		iter->second.m_nX = x;
		iter->second.m_nX = y;
	}

	//发送至UI处理
	ARRAY arr = new CArray();
	arr->Add(nOtherPlayerId);
	arr->Add(x);
	arr->Add(y);
	CSceneManager::sharedSceneManager()->PostMessageA(MSG_SC_OTHERPLAYER_RUN, arr, NULL);
}

void CGameNetBean::sc_otherplayer_join(STREAM stream)
{
	//处理游戏逻辑
	CGameModel *pModel = CGameModel::sharedGameModel();

	COtherPlayer otherPlayer;
	otherPlayer.m_nOtherPlayerID = stream.readInt();
	otherPlayer.m_strName = stream.readUTF();
	otherPlayer.m_nX = stream.readInt();
	otherPlayer.m_nY = stream.readInt();

	pModel->m_mOtherPlayerMap[otherPlayer.m_nOtherPlayerID] = otherPlayer;

	//发送至UI处理
	ARRAY arr = new CArray();
	arr->Add(otherPlayer.m_nOtherPlayerID);
	CSceneManager::sharedSceneManager()->PostMessageA(MSG_SC_OTHERPLAYER_JOIN, arr, NULL);
}

void CGameNetBean::sc_otherplayer_exit(STREAM stream)
{
	//处理游戏逻辑
	CGameModel *pModel = CGameModel::sharedGameModel();
	int nOtherPlayerId = stream.readInt();
	map<int,COtherPlayer>::iterator iter = pModel->m_mOtherPlayerMap.find(nOtherPlayerId);
	if(iter != pModel->m_mOtherPlayerMap.end())
	{
		pModel->m_mOtherPlayerMap.erase(iter);
	}

	//发送至UI处理
	ARRAY arr = new CArray();
	arr->Add(nOtherPlayerId);
	CSceneManager::sharedSceneManager()->PostMessageA(MSG_SC_OTHERPLAYER_EXIT, arr, NULL);
}

void CGameNetBean::cs_login(const char* nikename)
{
	STREAM stream;
	stream.writeInt(MSG_CS_LOGIN);
	stream.writeUTF(nikename);
	write(stream);
}

void CGameNetBean::cs_run(int x, int y)
{
	STREAM stream;
	stream.writeInt(MSG_CS_RUN);
	stream.writeInt(CGameModel::sharedGameModel()->m_Player.m_nPlayerID);
	stream.writeInt(x);
	stream.writeInt(y);
	write(stream);
}

void CGameNetBean::cs_exit()
{
	STREAM stream;
	stream.writeInt(MSG_CS_EXIT);
	stream.writeInt(CGameModel::sharedGameModel()->m_Player.m_nPlayerID);
	write(stream);
}
