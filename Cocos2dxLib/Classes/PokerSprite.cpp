#include "PokerSprite.h"


void PokerSprite::initFrame()
{

	CCSprite *number;
	CCSprite *color;
	/*CCSprite *bg=CCSprite::create(IMAGE_POKER_BG_DEFAULT);
	this->addChild(bg,10);*/


	if (m_number<14&&m_number>0)
	{
		switch (m_number)
		{
		case 1:case 2:case 3:case 4:case 5:case 6:case 7:case 8:case 9:case 10:case 11:
			number=(m_color%2)?CCSprite::create(IMAGE_POKER_NUMBER,CCRectMake(51*m_number+60,0,50,52)):CCSprite::create(IMAGE_POKER_NUMBER,CCRectMake(50*m_number+50,53,50,52));
			this->addChild(number,15,10003);
			break;
		case 12:case 13:
			number=(m_color%2)?CCSprite::create(IMAGE_POKER_NUMBER,CCRectMake(51*(m_number-12)+9,0,50,52)):CCSprite::create(IMAGE_POKER_NUMBER,CCRectMake(50*m_number-600,53,50,52));
			this->addChild(number,15,10003);
			break;
		default:
			break;
		}
		number->setPosition(ccp(30,100));
		switch (m_color)
		{
		case 1:
			color=CCSprite::create(IMAGE_POKER_COLOR,CCRectMake(102,0,34,37));
			this->addChild(color,15,10003);
			break;
		case 2:
			color=CCSprite::create(IMAGE_POKER_COLOR,CCRectMake(34,0,34,37));
			this->addChild(color,15,10003);
			break;
		case 3:
			color=CCSprite::create(IMAGE_POKER_COLOR,CCRectMake(68,0,34,37));
			this->addChild(color,15,10003);
			break;
		case 4:
			color=CCSprite::create(IMAGE_POKER_COLOR,CCRectMake(0,0,34,37));
			this->addChild(color,15,10003);
			break;
		default:
			break;
		}
		color->setPosition(ccp(30,140));

		return;
	}
	if (m_number==14)
	{
		number=CCSprite::create(IMAGE_POKER_JOKER,CCRectMake(0,0,40,166));
		this->addChild(number,15,10003);
		number->setPosition(ccp(30,90));
		return;
	}
	if (m_number==15)
	{
		number=CCSprite::create(IMAGE_POKER_JOKER,CCRectMake(41,0,40,166));
		this->addChild(number,15,10003);
		number->setPosition(ccp(30,90));
		return;
	}

}

void PokerSprite::onEnter()
{
	/*CCTouchDispatcher::addTargetedDelegate(this,-100,true);*/
	//this->setTouchEnabled(true);
	CCSprite::onEnter();
	this->registerWithTouchDispatcher();

}
void PokerSprite::onExit()
{
	CCDirector::sharedDirector()->getTouchDispatcher()->removeDelegate(this);
	CCSprite::onExit();
	//this->setTouchEnabled(false);


}
bool PokerSprite::ccTouchBegan(CCTouch *pTouch, CCEvent *pEvent)
{
	if (this->isVisible())
	{
		CCPoint tp=pTouch->getLocationInView();
		tp.y=CCDirector::sharedDirector()->getWinSize().height-tp.y;
		CCRect p_rect=	CCRectMake(
			this->getPosition().x-this->getContentSize().width*(this->getAnchorPoint().x), 
			this->getPosition().y-this->getContentSize().height*(this->getAnchorPoint().y),
			this->getContentSize().width,
			this->getContentSize().height);
		if( p_rect.containsPoint(tp))
		{
#ifdef TEST_OR_NOT
			CCLOG("touchCardBegan:%f,%f\n",tp.x,tp.y);			
#endif
			if(!this->m_selecting)
			{
				this->m_selecting=CCSprite::create(IMAGE_POKER_SELECTING);
				this->m_selecting->setPosition(ccp(this->getContentSize().width/2,this->getContentSize().height/2));
				this->addChild(this->m_selecting,20);
			}
			this->m_selecting->setVisible(true);
			return true;
		}
	}
	return false;

}
void PokerSprite::ccTouchMoved(CCTouch *pTouch, CCEvent *pEvent)
{
	if (this->isVisible())
	{
		if (m_manageothers!=0)
			PokerSprite::doingSelectionOthers(pTouch);
	}
}
void PokerSprite::ccTouchEnded(CCTouch *pTouch, CCEvent *pEvent)
{
	PokerSprite::endSelection();
}
void PokerSprite::doingSelectionOthers(CCTouch *pTouch)
{
	CCPoint tp=pTouch->getLocationInView();
	//tp.y=CCDirector::sharedDirector()->getWinSize().height-tp.y;
	int end=int(tp.x);
	int i;
	for ( i = 0; i < this->m_manageothers->length; i++)
	{
		if (*(m_manageothers->cardUIList+i)!=0)
		{
			end-=(int)((*(m_manageothers->cardUIList+i))->getPosition().x-(*(m_manageothers->cardUIList+i))->getContentSize().width*((*(m_manageothers->cardUIList+i))->getAnchorPoint().x));			
			end=end/m_manageothers->cardGap;
			end=end>=this->m_manageothers->length?this->m_manageothers->length-1:end;
			end=end<0?0:end;

			break;
		}
		//else i--;
	}
	tp=pTouch->getStartLocationInView();
	int begin=int(tp.x);
	for ( i = 0; i < this->m_manageothers->length; i++)//this->m_manageothers->length
	{
		if (*(m_manageothers->cardUIList+i)!=0)
		{
			begin-=(int)((*(m_manageothers->cardUIList+i))->getPosition().x-(*(m_manageothers->cardUIList+i))->getContentSize().width*((*(m_manageothers->cardUIList+i))->getAnchorPoint().x));
			begin=begin/m_manageothers->cardGap;
			begin=begin>=this->m_manageothers->length?this->m_manageothers->length-1:begin;
			begin=begin<0?0:begin;
			break;
		}
		//else i--;
	}
	if (begin>end)
	{
		int temp=begin;
		begin=end;
		end=temp;
	}
	//#ifdef TEST_OR_NOT
	//			CCLOG("BeganCard:%d EndCard%d\n",begin,end);			
	//#endif
	i=0;int j=begin;
	while (i<this->m_manageothers->length)
	{
		PokerSprite *temp=*(this->m_manageothers->cardUIList+i);
		if (temp!=0&&temp!=this)
		{
			if (i>=begin&&j<=end)
			{
				temp->doingSelection();
				j++;
			}
			else
			{
				temp->doingNoSelection();
			}

		}
		else if (temp==this) j++;
		i++;
	}

}

void PokerSprite::doingSelection()
{
	if(!this->m_selecting)
	{
		this->m_selecting=CCSprite::create(IMAGE_POKER_SELECTING);
		this->m_selecting->setPosition(ccp(this->getContentSize().width/2,this->getContentSize().height/2));
		this->addChild(this->m_selecting,20);
	}
	this->m_selecting->setVisible(true);
}
void PokerSprite::doingNoSelection()
{
	if(this->m_selecting)
	{
		this->m_selecting->setVisible(false);			
	}		
}
void PokerSprite::endSelection()
{
	if (this->m_manageothers!=0)
	{
		PokerSprite **begin=(this->m_manageothers->cardUIList);
		for (PokerSprite **p=begin; p<begin+this->m_manageothers->length; p++)
		{
			if (*p!=0&&(*p)->m_selecting &&(*p)->m_selecting->isVisible())
			{
				(*p)->m_selecting->setVisible(false);
				if ((*p)->m_selectStatus==2)
				{
					(*p)->m_selectStatus=0;
					(*p)->setPosition(ccp((*p)->getPosition().x,(*p)->getPosition().y-20));
				}
				else if ((*p)->m_selectStatus==0)
				{
					(*p)->m_selectStatus=2;
					(*p)->setPosition(ccp((*p)->getPosition().x,(*p)->getPosition().y+20));
				}
			}
		}
		CCLOG("touchCardEnded");
	}
	else
	{
		if (this->m_selecting &&this->m_selecting->isVisible())
		{
			this->m_selecting->setVisible(false);
			if (m_selectStatus==2)
			{
				m_selectStatus=0;
				this->setPosition(ccp(this->getPosition().x,this->getPosition().y-20));
			}
			else if (m_selectStatus==0)
			{
				m_selectStatus=2;
				this->setPosition(ccp(this->getPosition().x,this->getPosition().y+20));
			}
			CCLOG("touchCardEnded");
		}
	}


}