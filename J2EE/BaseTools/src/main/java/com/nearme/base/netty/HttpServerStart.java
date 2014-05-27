/**
 * HttpServerStart.java
 * com.nearme.base.netty
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-2-11 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty;

import java.util.concurrent.Executor;

import org.jboss.netty.channel.Channel;

import com.nearme.base.concurrent.ExecutorManager;
import com.nearme.base.netty.server.NettyHttpServer;

/**
 * ClassName:HttpServerStart <br>
 * Function: 模拟的http server启动 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-2-11  下午06:31:33
 */
public class HttpServerStart {
	public static Channel start(int port) {
		return start(port, true);
	}

	/**
	 * 开启一个接收http请求的netty服务
	 * @param port http服务端口
	 * @param asyncHandle 是否异步的处理请求
	 * @return
	 */
	public static Channel start(int port, boolean asyncHandle) {
		Executor handlerExecutor = null;
		if(asyncHandle) {
			handlerExecutor = ExecutorManager.newCachedThreadPool("Server-Handler-Executor");
		}

		return start(port, handlerExecutor);
	}

	public static Channel start(int port, final Executor handlerExecutor) {
		return start(port, true, true, null, null, handlerExecutor);
	}

	/**
	 * 开启一个接收http请求的netty服务
	 * @param port http服务端口
	 * @param tcpNoDelay
	 * @param reuseAddress
	 * @param bossExecutor
	 * @param workerExecutor
	 * @param handlerExecutor
	 * @return
	 */
	public static Channel start(int port, boolean tcpNoDelay, boolean reuseAddress,
			Executor bossExecutor, Executor workerExecutor, final Executor handlerExecutor) {
		NettyHttpServer server = new NettyHttpServer();

		return server.start(port, tcpNoDelay, reuseAddress, bossExecutor, workerExecutor, handlerExecutor);
	}
}

