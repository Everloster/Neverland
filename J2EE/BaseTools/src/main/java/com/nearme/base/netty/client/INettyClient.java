/**
 * INettyClient.java
 * com.nearme.base.netty.client
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-9-27 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.client;


/**
 * ClassName:INettyClient <br>
 * Function: netty client,构造请求对象RequestT，通过 getResponse获得返回对象ResponseD<br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-9-27  下午05:40:31
 */
public interface INettyClient<RequestT, ResponseD> {
	/**
	 * 获取对应的标识
	 * @param 
	 * @return
	 */
	public abstract String getIdentifier();
	
	/**
	 * 从服务端获取数据
	 * @param request request数据
	 * @return
	 */
	public abstract ResponseD getResponse(RequestT request) throws Exception;
	
	/**
	 * 从服务端获取数据
	 * @param request request数据
	 * @param additional 额外的数据
	 * @return
	 */
	public abstract ResponseD getResponse(RequestT request, Object additional) throws Exception;
	
	/**
	 * 接收请求
	 */
	public abstract void putResponse(ResponseD response) throws Exception;
	
	/**
	 * 停止连接服务
	 * @param 
	 * @return
	 */
	public abstract void stop();
}

