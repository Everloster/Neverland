/**
 * IBusinessServerFactory.java
 * com.nearme.base.netty.connect
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-12-7 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/
/**
 * Copyright (c) 2012 NearMe, All Rights Reserved.
 * FileName:IBusinessServerFactory.java
 * ProjectName:NearmeBaseToolsJ
 * PackageName:com.nearme.base.netty.connect
 * Create Date:2012-12-7
 * History:
 *   ver	 date	       author		desc
 * ────────────────────────────────────────────────────────
 *   1.0	2012-12-7	  	80036381		
 *
 * 
 */


package com.nearme.base.netty.connect;

import com.nearme.base.concurrent.MultiCPUTask;
import com.nearme.base.netty.client.INettyClient;

/**
 * ClassName:IBusinessServerFactory <br>
 * Function: 业务服务连接对象生成工厂 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-12-7  下午02:12:03
 */
public interface IBusinessServerFactory<T, D> {
	/**
	 * 默认的客户端连接服务端的超时时间(建立连接)
	 */
	public static final int DEFAULT_CLIENT_TIMEOUT = 120 * 1000;
	/**
	 * 默认的客户端连接个数
	 */
	public static final int DEFAULT_CLIENT_COUNT = MultiCPUTask.DEFAULT_THREAD_COUNT;
	
	/**
	 * 获取连接核心业务服务的客户端
	 * @param 
	 * @return
	 */
	public INettyClient<T, D> getConnectClient() throws Exception;
	
	/**
	 * 获取核心业务服务的所有客户端连接
	 * @param 
	 * @return
	 */
	public INettyClient<T, D>[] getConnectClients() throws Exception;
	
	/**
	 * 加载服务器列表
	 * @param 
	 * @return
	 */
	public abstract BusinessServerInfo[] loadServerInfo();
	
	/**
	 * 从多个服务器中选择一个
	 * @param 
	 * @return
	 */
	public abstract BusinessServerInfo select(BusinessServerInfo[] servers);
}

