package nio.game.server;

import static nio.game.utils.LOG.*;

import nio.game.logic.GameLogicBean;
import nio.game.utils.LOG;

import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.channel.ChannelStateEvent;
import org.jboss.netty.channel.ExceptionEvent;
import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.channel.SimpleChannelUpstreamHandler;

public class GameLogicHandler extends SimpleChannelUpstreamHandler{

	/******************************************************
	* 功能描述：当有流消息时触发调用,在这个方法里进行数据解码，
	* 			消息号解析，消息分发，业务处理
	*******************************************************/
	@Override
	public void messageReceived(ChannelHandlerContext ctx, MessageEvent e) throws Exception {
		//获取客户端业务流程处理器
		GameLogicBaseBean nnLogicBean = GameLogicContext.sharedGameContext().GetClient(e.getChannel());		
		//如果有意外情况，导致客户端处理器已经不存在，就释放客户端
		if(nnLogicBean == null){
			E("if(cl != null){");
			GameLogicContext.sharedGameContext().RemoveClient(e.getChannel());
			super.messageReceived(ctx, e);
			return;
		}
		//通知处理器，读到新的数据
		nnLogicBean.OnMessage((ChannelBuffer) e.getMessage());
	}
	
	/******************************************************
	* 功能描述：客户端连接时调用
	*******************************************************/
	@Override
	public void channelConnected(ChannelHandlerContext ctx, ChannelStateEvent e) throws Exception {
		// >>
		Channel nnChannel = e.getChannel();
		//创建客户端逻辑处理器
		GameLogicBaseBean nnLogicBean = new GameLogicBean(nnChannel);
		//加入客户端列表
		GameLogicContext.sharedGameContext().AddClient(nnChannel, nnLogicBean);
		//通知逻辑层已经连接成功
		nnLogicBean.OnConnected();
		// <<
		super.channelConnected(ctx, e);
	}

	/******************************************************
	* 功能描述：客户端断线时调用
	*******************************************************/
	@Override
	public void channelDisconnected(ChannelHandlerContext ctx, ChannelStateEvent e) throws Exception {
		// >>
		//获取客户端业务流程处理器
		GameLogicBaseBean nnLogicBean = GameLogicContext.sharedGameContext().GetClient(e.getChannel());
		if (nnLogicBean != null) {
			//通知处理器已经断线，保存数据或其他操作
			nnLogicBean.OnDisconnected();
		} else {
			E("channelDisconnected if(nnLogicBean != null){");
		}
		//移除客户端并关闭通道
		GameLogicContext.sharedGameContext().RemoveClient(e.getChannel());
		// <<
		
		LOG.D("客户端IO总数:"+ GameLogicContext.sharedGameContext().GetClientChannelGroupSize()
				+ "  " + "客户端逻辑处理器总数:" + GameLogicContext.sharedGameContext().GetClientSize()
				+ "  " + "客户端数据集合总数:" + GameLogicContext.sharedGameContext().GetGameDataSize());
		
		super.channelDisconnected(ctx, e);
	}


	/******************************************************
	* 功能描述：内部IO读写发生异常时调用
	*******************************************************/
	@Override
	public void exceptionCaught(ChannelHandlerContext ctx, ExceptionEvent e) throws Exception {
		// >>
		//获取客户端业务流程处理器
		GameLogicBaseBean nnLogicBean = GameLogicContext.sharedGameContext().GetClient(e.getChannel());
		if (nnLogicBean != null) {
			//通知处理器已经断线，保存数据或其他操作
			nnLogicBean.OnDisconnected();
		} else {
			E("exceptionCaught if(nnLogicBean != null){");
		}
		//移除客户端并关闭通道
		GameLogicContext.sharedGameContext().RemoveClient(e.getChannel());
		
		LOG.D("客户端IO总数:"+ GameLogicContext.sharedGameContext().GetClientChannelGroupSize()
				+ "  " + "客户端逻辑处理器总数:" + GameLogicContext.sharedGameContext().GetClientSize()
				+ "  " + "客户端数据集合总数:" + GameLogicContext.sharedGameContext().GetGameDataSize());
		// <<
		super.exceptionCaught(ctx, e);
	}
	
	

	
}
