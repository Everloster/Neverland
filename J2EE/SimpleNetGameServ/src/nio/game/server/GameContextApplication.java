package nio.game.server;

import static org.jboss.netty.channel.Channels.*;

import java.net.InetSocketAddress;
import java.util.concurrent.Executors;

import org.jboss.netty.bootstrap.ServerBootstrap;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelPipeline;
import org.jboss.netty.channel.ChannelPipelineFactory;
import org.jboss.netty.channel.socket.nio.NioServerSocketChannelFactory;
import org.jboss.netty.handler.codec.frame.LengthFieldPrepender;

public class GameContextApplication{
	
	/******************************************************
	* 功能描述：服务器
	*******************************************************/
	private ServerBootstrap mServerBoot = null;
	/******************************************************
	* 功能描述：使用NIO无阻塞通道工厂来创建通道
	*******************************************************/
	private NioServerSocketChannelFactory mNioChannelFactory = null;
	/******************************************************
	* 功能描述：Server在创建时产生的Channel对象，结束时释放
	*******************************************************/
	@SuppressWarnings("unused")
	private Channel mServerChannel = null;
	
	

	/******************************************************
	* 功能描述：启动游戏服务器，初始化所有资源
	*******************************************************/
	public void run(){
		
		mNioChannelFactory = new NioServerSocketChannelFactory(
				Executors.newCachedThreadPool(),
				Executors.newCachedThreadPool());
		
		mServerBoot = new ServerBootstrap(mNioChannelFactory);
		
		//创建每一个客户端需要使用的管道
		mServerBoot.setPipelineFactory(new ChannelPipelineFactory() {
			@Override
			public ChannelPipeline getPipeline() throws Exception {
				//创建管道，并添加管道拦截器
				ChannelPipeline nnPipeline = pipeline();
//				nnPipeline.addLast("decoder", new LengthFieldBasedFrameDecoder(1024, 0, 4, 0, 4 ,true));
				nnPipeline.addLast("decoder", new GameFrameDecoder());
				nnPipeline.addLast("handler", new GameLogicHandler());
				nnPipeline.addLast("encoder", new LengthFieldPrepender(4, false));
				return nnPipeline;
			}
		});
		
		//绑定端口，并开启监听
		mServerChannel = mServerBoot.bind(new InetSocketAddress(7789));
	}
	
	
	public static void main(String[] args) {
		GameContextApplication application = new GameContextApplication();
		application.run();
	}
}
