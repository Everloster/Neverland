#ifndef __PLAYER_H__
#define __PLAYER_H__

#include "cocos2d.h"
using namespace cocos2d;

//玩家对象
class CPlayerSprite : public CCSprite
{
public:
	bool init(const char* pszFileName, const char* pName);
	void run(int nX, int nY);

public:
	static CPlayerSprite* create(const char* pName);

private:
	CCLabelTTF *pNameLabel;
};

#endif //__PLAYER_H__