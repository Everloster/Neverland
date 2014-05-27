package nio.game.server;

import java.util.Collection;

import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.util.internal.ConcurrentHashMap;

public class GameLogicContext {
	
	//静态单例实例
	private final static GameLogicContext mGameLogicContext = new GameLogicContext();
	
	/******************************************************
	* 功能描述：静态单例模式，获取GAME游戏逻辑的实例对象
	*******************************************************/
	public static GameLogicContext sharedGameContext(){
		return mGameLogicContext;
	}
	
	/* IO组
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	//游戏所有客户端的IO读写列表（所有玩家都在此列表里）
	private final ConcurrentHashMap<Integer, Channel> mClientChannelGroup = new ConcurrentHashMap<Integer, Channel>();
	
	// >> 如果要对客户端IO所在列表进行分段，在此定义，便于遍历
	
	
	// <<
	
	
	/* 逻辑组
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	//游戏所有客户端对应的游戏逻辑处理器
	private final ConcurrentHashMap<Integer, GameLogicBaseBean> mClientLogicGroup = new ConcurrentHashMap<Integer, GameLogicBaseBean>();
	

	
	/* 数据组
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	//游戏所有客户端对应的客户端数据
	private final ConcurrentHashMap<Integer, GameLogicBaseData> mClientDataGroup = new ConcurrentHashMap<Integer, GameLogicBaseData>();
	

	/* Common
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	//加入一个新的客户端逻辑处理器
	public void AddClient(Channel nnChannel, GameLogicBaseBean nnGameBean){
		int nnID = nnChannel.getId();
		this.mClientLogicGroup.put(nnID, nnGameBean);
		this.mClientChannelGroup.put(nnID, nnChannel);
	}

	//删除一个客户端逻辑处理器，并关掉IO通道
	public void RemoveClient(Channel nnChannel){
		int nnID = nnChannel.getId();
		this.mClientLogicGroup.remove(nnID);
		this.mClientChannelGroup.remove(nnID);
		nnChannel.close();
	}

	//获得一个客户端逻辑处理器
	public GameLogicBaseBean GetClient(Channel nnChannel){
		return this.mClientLogicGroup.get(nnChannel.getId());
	}
	
	//获得客户端逻辑处理器总数量
	public int GetClientSize(){
		return this.mClientLogicGroup.size();
	}
	
	//获得所有客户端IO集合
	public ConcurrentHashMap<Integer, Channel> GetClientChannelGroup(){
		return this.mClientChannelGroup;
	}
	
	//获得所有客户端IO集合的个数
	public int GetClientChannelGroupSize(){
		return this.mClientChannelGroup.size();
	}
	
	//加一个新的客户端数据
	public void AddGameData(int nnID, GameLogicBaseData data){
		mClientDataGroup.put(nnID, data);
	}
	
	//删除某一个客户端的数据
	public void RemoveGameData(int nnID){
		mClientDataGroup.remove(nnID);
	}
	
	//获取某一个客户端的数据
	public GameLogicBaseData GetGameData(int nnID){
		return mClientDataGroup.get(nnID);
	}
	
	//获取客户端数据集的个数
	public int GetGameDataSize(){
		return mClientDataGroup.size();
	}
	
	//获取客户端总数
	public int size(){
		return this.mClientDataGroup.size();
	}
	
	//需要广播数据给服务器其他所有玩家
	public void write(ChannelBuffer nnChannelBuffer){		
		Collection<Channel> values = mClientChannelGroup.values();
		for (Channel nnChannel: values) {
			nnChannel.write(nnChannelBuffer);
        }
	}

	//某一个玩家需要广播数据给服务器其他所有玩家，但是不广播给自己 nnChannelID用来排除自己
	public void write(ChannelBuffer nnChannelBuffer, int nnChannelID){
		Collection<Channel> values = mClientChannelGroup.values();
		for (Channel nnChannel: values) {
			if(nnChannel.getId() != nnChannelID){
				nnChannel.write(nnChannelBuffer);
			}
        }
	}
}
