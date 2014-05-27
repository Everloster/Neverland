#ifndef __HTTP_CLIENT_H__
#define __HTTP_CLIENT_H__

#include <stdio.h>
#include <stdlib.h>
#include "cocos2d.h"
#include "cocos-ext.h"

USING_NS_CC;
USING_NS_CC_EXT;

class CHttpServer
{
private:
	//构造函数
	CHttpServer(void);
	//析构函数
	virtual ~CHttpServer(void);
public:
	//发送请求
	virtual void send(CCHttpRequest* request);
public:
	//单利模式
	static CHttpServer* sharedHttpServer(void);
	//通过url获取Get对象
	static CCHttpRequest* getGetRequestByUrl(const char* pUrl);
	//通过url和data获取Post对象
	static CCHttpRequest* getPostRequestByUrlAndData(const char* pUrl, const char* pData);
private:
	//处理默认的请求头
	static void addDefaultHeaders(CCHttpRequest* pRequest);
};

#endif