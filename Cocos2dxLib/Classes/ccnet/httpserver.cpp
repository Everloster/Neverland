#include "httpserver.h"

CHttpServer::CHttpServer(void)
{

}

CHttpServer::~CHttpServer(void)
{

}

CHttpServer* CHttpServer::sharedHttpServer()
{
	static CHttpServer* pInstance = NULL;

	if (pInstance == NULL)
	{
		pInstance = new CHttpServer();
	}
	return pInstance;
}

CCHttpRequest* CHttpServer::getGetRequestByUrl(const char* pUrl)
{
	CCHttpRequest* pRequest = new CCHttpRequest();
	pRequest->setUrl(pUrl);
	pRequest->setRequestType(CCHttpRequest::kHttpGet);
	addDefaultHeaders(pRequest);
	return pRequest;
}

CCHttpRequest* CHttpServer::getPostRequestByUrlAndData(const char* pUrl, const char* pData)
{
	CCHttpRequest* pRequest = new CCHttpRequest();
	pRequest->setUrl(pUrl);
	pRequest->setRequestType(CCHttpRequest::kHttpPost);
	pRequest->setRequestData(pData ,strlen(pData));
	addDefaultHeaders(pRequest);
	return pRequest;
}


void CHttpServer::addDefaultHeaders(CCHttpRequest* pRequest)
{
	std::vector<std::string> headers;  
	headers.push_back("Content-Type: application/json;charset=utf-8");  
	pRequest->setHeaders(headers);
}

void CHttpServer::send(CCHttpRequest* request)
{
	static CCHttpClient* pClient = NULL;
	if (pClient == NULL) 
	{
		CCLog("create CCHttpClient instance in CHttpServer.");
		pClient = CCHttpClient::getInstance();
	}
	return pClient->send(request);
}

