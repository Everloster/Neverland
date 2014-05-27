#ifndef __GAMENETBEAN_H__
#define __GAMENETBEAN_H__

#include "ccnet/net.h"
#include "ccbase/base.h"
#include "GameModel.h"

#define MSG_UI_CONNECTED 1					//通知UI连接成功
#define MSG_UI_DISCONNECTED 2				//通知UI连接中断
#define MSG_UI_CONNECTERROR	3				//通知UI连接错误
#define MSG_UI_CONNECTTIMEOUT 4				//通知UI连接超时

///cs接口消息号
#define MSG_CS_LOGIN 1000					//登陆
#define MSG_CS_RUN	1001					//跑步
#define MSG_CS_EXIT	1002					//离开房间

///sc接口消息号
#define MSG_SC_ERROR 5000					//错误消息
#define MSG_SC_INITCLIENT 5001				//初始化客户端
#define MSG_SC_OTHERPLAYER_RUN 5002			//其他玩家跑步
#define MSG_SC_OTHERPLAYER_JOIN	5003		//其他玩家加入房间
#define MSG_SC_OTHERPLAYER_EXIT 5004		//其他玩家离开房间

///错误号



class CGameNetBean : public CNetBean, public CLooper
{
public:
	DEFINE_SINGLE_FUNCTION(CGameNetBean);

public:
	virtual ~CGameNetBean();

public:
	///当创建成功 开始连接时调用
	virtual void onCreate();
	///当连接成功时调用
	virtual void onConnected();
	///当断开连接时调用
	virtual void onDisconnected();
	///当连接错误时调用
	virtual void onConnectError();
	///当连接超时时调用
	virtual void onConnectTimeout();
	///当读到消息时调用
	virtual void onMessage(STREAM stream);

public:
	//每帧回调方法
	virtual void loop() { this->drive(); }



//sc服务端发给客户端的回调接口
public:
	//游戏里所有的错误信息返回接口
	//参数：[int:消息号] [int:错误号]
	void sc_error(STREAM stream);

	//登陆成功后调用，通知客户端初始化数据
	//参数：[int:消息号] [int:玩家的id号] [str:玩家的昵称] [int:玩家的初始x坐标] [int:玩家的初始y坐标]
	//[int:$count房间内已存在的玩家数量]
	//$count * {[int:其他玩家id] [str:其他玩家名称] [int:其他玩家x坐标] [int:其他玩家y坐标]}
	void sc_initclient(STREAM stream);

	//其他玩家跑步了
	//参数：[int:消息号] [int:其他玩家id] [int:x坐标] [int:y坐标]
	void sc_otherplayer_run(STREAM stream);

	//其他玩家进入房间了
	//参数：[int:消息号] [int:其他玩家id] [str:其他玩家名称] [int:其他玩家的x坐标] [int:其他玩家的y坐标]
	void sc_otherplayer_join(STREAM stream);

	//其他玩家离开房间
	//参数：[int:消息号] [int:其他玩家id]
	void sc_otherplayer_exit(STREAM stream);


//cs客户端发给服务端的接口
public:
	//请求登陆，如果没有用户名，默认是注册，登陆之后就进入房间了
	//参数：[int:消息号] [str:自定义的名称]
	//返回：[ok:sc_initclient] [no:sc_error]
	void cs_login(const char* nikename);

	//玩家跑步。通知房间内其他玩家
	//参数：[int:消息号] [int:玩家id] [int:x坐标] [int:y坐标]
	void cs_run(int x, int y);

	//请求离开房间，通知房间内其他玩家
	//参数：[int:消息号] [int:玩家id]
	void cs_exit();

};





#endif //__GAMENETBEAN_H__