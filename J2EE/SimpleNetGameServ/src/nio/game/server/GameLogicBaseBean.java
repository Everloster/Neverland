package nio.game.server;

import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.channel.Channel;

public abstract class GameLogicBaseBean {
	
	//客户端对应Channel的ID号
	protected int m_nnID;
	//客户端IOChannel
	protected Channel m_nnChannel;
	//客户端数据
	protected GameLogicBaseData m_GameLogicData;
	
	//是否是合法客户端
	//public boolean m_bIsIllegal;
	//连接上服务器的时间
	//public long m_nConnectedTime;
	
	/******************************************************
	* 功能描述：构造方法，初始化一个客户端IO通道
	*******************************************************/
	public GameLogicBaseBean(Channel nnChannel){
		this.m_nnChannel = nnChannel;
		this.m_nnID = nnChannel.getId();
	}
	/******************************************************
	* 功能描述：生成一个新的游戏逻辑数据对象。
	*******************************************************/
	public void registNewGameData(GameLogicBaseData nnGameLogicData){
		this.m_GameLogicData = nnGameLogicData;
		this.m_GameLogicData.OnCreate();
		GameLogicContext.sharedGameContext().AddGameData(this.m_GameLogicData.m_nnID, this.m_GameLogicData);
	}
	/******************************************************
	* 功能描述：删除游戏逻辑数据对象。
	*******************************************************/
	public void releaseThisGameData(){
		if(this.m_GameLogicData != null){
			GameLogicContext.sharedGameContext().RemoveGameData(this.m_GameLogicData.m_nnID);
			this.m_GameLogicData.OnDestroy();
			this.m_GameLogicData = null;
		}
	}
	/******************************************************
	* 功能描述：往对应客户端IO通道写入数据
	*******************************************************/
	public void wrtie(ChannelBuffer nnChannelBuffer){
		m_nnChannel.write(nnChannelBuffer);
	}
	/******************************************************
	* 功能描述：当有流消息时触发调用,在这个方法里进行，消息号解析，消息分发，业务处理
	*******************************************************/
	public abstract void OnMessage(ChannelBuffer nnChannelBuffer);
	/******************************************************
	* 功能描述：当客户端连接到服务器后，调用用于初始化数据
	*******************************************************/
	public abstract void OnConnected();
	/******************************************************
	* 功能描述：与客户端失去连接时调用
	*******************************************************/
	public abstract void OnDisconnected();
	

}
