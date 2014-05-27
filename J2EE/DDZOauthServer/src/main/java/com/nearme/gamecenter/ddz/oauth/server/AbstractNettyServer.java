/**
 * AbstractNettyServer.java
 * com.nearme.base.netty.server
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-4-26 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.gamecenter.ddz.oauth.server;

import java.net.InetSocketAddress;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executor;

import org.jboss.netty.bootstrap.Bootstrap;
import org.jboss.netty.bootstrap.ServerBootstrap;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelPipelineFactory;
import org.jboss.netty.channel.socket.nio.NioServerSocketChannelFactory;

import com.nearme.gamecenter.ddz.oauth.tool.concurrent.ExecutorManager;
import com.nearme.gamecenter.ddz.oauth.tool.concurrent.NamedThreadFactory;


/**
 * ClassName:AbstractNettyServer <br>
 * Function: 抽象的netty server <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-4-26  下午09:49:21
 */
public abstract class AbstractNettyServer {

	private Map<String, Object> options = new HashMap<String, Object>();
	private boolean isKeepAlive;

	/**
	 * 设置启动选项
	 * @param
	 * @return
	 */
	public void setOption(String key, Object value) {
		if (key == null) {
            throw new NullPointerException("key");
        }

		boolean isKeyForkeepAlive = key.equals("child.keepAlive");
        if (value == null) {
            options.remove(key);

            if(isKeyForkeepAlive) {
            	this.setKeepAlive(false);
            }
        } else {
            options.put(key, value);

            if(isKeyForkeepAlive) {
            	boolean keepAlive = false;
            	if(value instanceof String) {
            		keepAlive = Boolean.parseBoolean((String)value);
            	} else if(value instanceof Boolean) {
            		keepAlive = (Boolean)value;
            	}

            	this.setKeepAlive(keepAlive);
            }
        }
	}

	protected boolean isKeepAlive() {
		return isKeepAlive;
	}

	/**
	 * 设置isKeepAlive
	 * @param   isKeepAlive
	 * @since   Ver 1.0
	 */
	protected void setKeepAlive(boolean isKeepAlive) {
		this.isKeepAlive = isKeepAlive;
	}

	public Channel start(int port) {
		return start(port, true);
	}

	/**
	 * 开启一个接收请求的netty服务
	 * @param port 服务端口
	 * @param asyncHandle 是否异步的处理请求
	 * @return
	 */
	public Channel start(int port, boolean asyncHandle) {
		Executor handlerExecutor = null;
		if(asyncHandle) {
//			handlerExecutor = ExecutorManager.newCachedThreadPool("Server-Handler-Executor");
			handlerExecutor = ExecutorManager.newFixedThreadPool(50, new NamedThreadFactory("Server-Handler-Executor"));
		}

		return start(port, handlerExecutor);
	}

	public Channel start(int port, final Executor handlerExecutor) {
		return start(port, true, true, null, null, handlerExecutor);
	}

	/**
	 * 开启一个接收请求的netty服务
	 * @param port 服务端口
	 * @param tcpNoDelay
	 * @param reuseAddress
	 * @param bossExecutor
	 * @param workerExecutor
	 * @param handlerExecutor
	 * @return
	 */
	public Channel start(int port, boolean tcpNoDelay, boolean reuseAddress,
			Executor bossExecutor, Executor workerExecutor, Executor handlerExecutor) {
		//如果不存在则创建工作线程池
		if(null == bossExecutor) {
			bossExecutor = ExecutorManager.newCachedThreadPool("Server-Boss-Executor");
		}

		if(null == workerExecutor) {
			workerExecutor = ExecutorManager.newCachedThreadPool("Server-Worker-Executor");
		}

		//final Timer timer = new HashedWheelTimer();	//负责超时管理

		//配置服务器-使用java线程池作为解释线程
		ServerBootstrap bootstrap = new ServerBootstrap(
				new NioServerSocketChannelFactory(bossExecutor, workerExecutor));
		//这两个参数对优化很重要，因此直接写在这里了
		bootstrap.setOption("child.tcpNoDelay", tcpNoDelay);
		//reuseAddress选项设置为true将允许将套接字绑定到已在使用中的地址
		bootstrap.setOption("reuseAddress", reuseAddress);

		if(!options.isEmpty()) {
			for(String key : options.keySet()) {
				bootstrap.setOption(key, options.get(key));
			}
		}

		// 设置 pipeline factory.
		bootstrap.setPipelineFactory(getChannelPipelineFactory(handlerExecutor));

		// 绑定端口
		if(port < NettyConstants.MIN_SERVER_PORT || port > NettyConstants.MAX_SERVER_PORT) {
			port = NettyConstants.DEFAULT_SERVER_PORT;
		}

		addShutdownHook(bootstrap);

		return bootstrap.bind(new InetSocketAddress(port));
	}

	protected void addShutdownHook(final Bootstrap bootstrap) {
//		final AbstractNettyServer self = this;
//		Runtime.getRuntime().addShutdownHook(new Thread(){
//			@Override
//			public void run() {
//				bootstrap.releaseExternalResources();
//				LoggerFactory.getLogger(self.getClass()).info("系统释放资源完成");
//			}
//		});
	}

	/**
	 * 获取一个ChannelPipelineFactory
	 * @param handlerExecutor
	 * @return
	 */
	public abstract ChannelPipelineFactory getChannelPipelineFactory(Executor handlerExecutor);
}

