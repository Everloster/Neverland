#include "OppoServer.h"


OppoServer::OppoServer()
{
	m_pServerDelegate = CHttpServer::sharedHttpServer();
}

OppoServer::~OppoServer()
{

}

OppoServer* OppoServer::shareOppoServer()
{
	static OppoServer* pServer = NULL;
	if (pServer == NULL) 
	{
		pServer = new OppoServer();
	}
	return pServer;
}

void OppoServer::send(CCHttpRequest* pRequest)
{
	m_pServerDelegate->send(pRequest);
}

void OppoServer::requestLogin()
{

}