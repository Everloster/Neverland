package nio.game.logic;

import java.io.UnsupportedEncodingException;
import java.nio.charset.Charset;
import java.util.Collection;
import java.util.Iterator;

import net.game.pojo.GameData;
import nio.game.server.GameContextApplication;
import nio.game.server.GameLogicBaseBean;
import nio.game.server.GameLogicContext;
import nio.game.utils.LOG;

import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.buffer.ChannelBuffers;
import org.jboss.netty.channel.Channel;

public class GameLogicBean extends GameLogicBaseBean{

	///cs接口消息号
	public static final int MSG_CS_LOGIN = 1000;					//登陆
	public static final int MSG_CS_RUN	= 1001;					//跑步
	public static final int MSG_CS_EXIT	= 1002;					//离开房间

	///sc接口消息号
	public static final int MSG_SC_ERROR = 5000;					//错误消息
	public static final int MSG_SC_INITCLIENT = 5001;				//初始化客户端
	public static final int MSG_SC_OTHERPLAYER_RUN = 5002;			//其他玩家跑步
	public static final int MSG_SC_OTHERPLAYER_JOIN	= 5003;		//其他玩家加入房间
	public static final int MSG_SC_OTHERPLAYER_EXIT = 5004;		//其他玩家离开房间
	
	///错误号
	public static final int ERRNO_LOGIN_PASSWORD = 10000;			//登陆错误，密码不正确
	
	//Data center
	private GameData GAMEDATA;
	
	public GameLogicBean(Channel nnChannel) {
		super(nnChannel);
	}

	@Override
	public void OnConnected() {
		LOG.D("客户端IO总数:"+ GameLogicContext.sharedGameContext().GetClientChannelGroupSize()
				+ "  " + "客户端逻辑处理器总数:" + GameLogicContext.sharedGameContext().GetClientSize()
				+ "  " + "客户端数据集合总数:" + (GameLogicContext.sharedGameContext().GetGameDataSize() + 1));
	}

	@Override
	public void OnDisconnected() {
		ChannelBuffer buff = ChannelBuffers.buffer(8);
		buff.writeInt(MSG_SC_OTHERPLAYER_EXIT);
		buff.writeInt(this.m_nnID);
		GameLogicContext.sharedGameContext().write(buff, this.m_nnID);
		this.releaseThisGameData();
	}
	
	
	@Override
	public void OnMessage(ChannelBuffer nnChannelBuffer) {
		
		switch(nnChannelBuffer.readInt())
		{
		case MSG_CS_LOGIN:
			cs_login(nnChannelBuffer);
			break;
		case MSG_CS_RUN:
			cs_run(nnChannelBuffer);
			break;
		}
	}
	
	void cs_run(ChannelBuffer nnChannelBuffer) {
		int nID = nnChannelBuffer.readInt();
		int x = nnChannelBuffer.readInt();
		int y = nnChannelBuffer.readInt();
		
		this.GAMEDATA.m_nX = x;
		this.GAMEDATA.m_nY = y;
		
		ChannelBuffer buff = ChannelBuffers.buffer(16);
		buff.writeInt(MSG_SC_OTHERPLAYER_RUN);
		buff.writeInt(this.m_nnID);
		buff.writeInt(x);
		buff.writeInt(y);
		GameLogicContext.sharedGameContext().write(buff, this.m_nnID);
	}
	
	void cs_login(ChannelBuffer nnChannelBuffer) {
		
		int username_length = nnChannelBuffer.readInt();
		String userName = nnChannelBuffer.toString(8, username_length, Charset.forName("UTF8"));
		
		this.GAMEDATA = new GameData(this.m_nnID);
		this.GAMEDATA.m_strName = userName;
		this.GAMEDATA.m_nX = 200;
		this.GAMEDATA.m_nY = 200;
		this.GAMEDATA.m_nnID = this.m_nnID;
		
		ChannelBuffer buff = ChannelBuffers.dynamicBuffer();
		buff.writeInt(MSG_SC_INITCLIENT);
		buff.writeInt(this.m_nnID);
		byte[] strbytes = null;
		try {
			strbytes = userName.getBytes("UTF8");
		} catch (UnsupportedEncodingException e1) {
			e1.printStackTrace();
		}
		buff.writeInt(strbytes.length);
		buff.writeBytes(strbytes);
		buff.writeInt(this.GAMEDATA.m_nX);
		buff.writeInt(this.GAMEDATA.m_nY);
		buff.writeInt(GameLogicContext.sharedGameContext().GetGameDataSize());
		
		ChannelBuffer tbuf = ChannelBuffers.dynamicBuffer();
		tbuf.writeInt(MSG_SC_OTHERPLAYER_JOIN);
		tbuf.writeInt(this.m_nnID);
		tbuf.writeInt(strbytes.length);
		tbuf.writeBytes(strbytes);
		tbuf.writeInt(this.GAMEDATA.m_nX);
		tbuf.writeInt(this.GAMEDATA.m_nY);
		
		Collection<Channel> clientChannels = GameLogicContext.sharedGameContext().GetClientChannelGroup().values();
		for(Channel channel : clientChannels){
			if(channel.getId() != this.m_nnID){
				GameData nnGameData = (GameData)GameLogicContext.sharedGameContext().GetGameData(channel.getId());
				buff.writeInt(nnGameData.m_nnID);
				byte[] names = null;
				try {
					names = nnGameData.m_strName.getBytes("UTF8");
				} catch (UnsupportedEncodingException e) {
					e.printStackTrace();
				}
				buff.writeInt(names.length);
				buff.writeBytes(names);
				buff.writeInt(nnGameData.m_nX);
				buff.writeInt(nnGameData.m_nY);
	
				channel.write(tbuf);
			}
		}
		
		
		this.registNewGameData(this.GAMEDATA);
		this.wrtie(buff);
	}
}
