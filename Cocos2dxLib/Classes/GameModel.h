#ifndef __GAMEMODEL_H__
#define __GAMEMODEL_H__

#include <map>
using namespace std;

class CPlayer;
class COtherPlayer;
class CGameModel;

//其他玩家的数据结构
class COtherPlayer
{
public:
	int m_nOtherPlayerID;	//其他玩家ID
	string m_strName;		//玩家昵称
	int m_nX;				//坐标
	int m_nY;
};

//主角玩家的数据结构
class CPlayer
{
public:
	int m_nPlayerID;		//玩家ID
	string m_strName;		//玩家昵称
	int m_nX;				//坐标
	int m_nY;
};

//游戏数据中心
class CGameModel
{
public:
	//主玩家
	CPlayer m_Player;
	//房间里的其他玩家列表
	map<int, COtherPlayer> m_mOtherPlayerMap;

public:
	//单例
	static CGameModel* sharedGameModel()
	{
		static CGameModel* model = NULL;
		if(model == NULL)
		{
			model = new CGameModel();
		}
		return model;
	};
};




#endif //__GAMEMODEL_H__