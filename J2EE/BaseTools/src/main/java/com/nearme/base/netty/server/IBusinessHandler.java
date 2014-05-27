/**
 * IBusinessHandler.java
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

package com.nearme.base.netty.server;

import org.jboss.netty.channel.MessageEvent;

import com.nearme.base.netty.common.ProtobufWrapperProtocol.RequestData;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.ResponseData;

/**
 * ClassName:IBusinessHandler <br>
 * Function: 业务处理接口 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-4-26  下午11:52:34
 */
public interface IBusinessHandler {
	/**
	 * 处理request请求
	 * @param request
	 * @param e
	 * @return
	 */
	public ResponseData handleRequest(RequestData request, MessageEvent e);
	
	/**
	 * 是否是异步的业务处理，如果是则统计信息、数据输出由业务自己控制，否则交给NettyBusinessServerHandler统一处理
	 * @param 
	 * @return
	 */
	public boolean isAsync();
}

