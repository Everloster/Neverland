#ifndef __OPPO_SERVER__
#define __OPPO_SERVER__

#include "cocos2d.h"
#include "ccnet/httpserver.h"

class OppoServer
{
private:
	OppoServer();
	~OppoServer();
	
public:

	//发送请求
	virtual void send(CCHttpRequest* pRequest);
	//请求登陆 TODO
	virtual void requestLogin();

private:
	//代理Server
	CHttpServer* m_pServerDelegate;

public:
	//单例模式
	static OppoServer* shareOppoServer();
};




#endif