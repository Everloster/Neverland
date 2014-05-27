/**
 * IRequestHandler.java
 * com.nearme.base.netty.server
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-1-29 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.gamecenter.ddz.oauth.server;

import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.handler.codec.http.HttpRequest;
import org.jboss.netty.handler.codec.http.HttpResponse;

import com.nearme.gamecenter.ddz.oauth.common.RequestConfig;


/**
 * ClassName:IRequestHandler <br>
 * Function: 请求处理接口，负责处理netty server转发过来的请求 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-29  下午02:14:42
 */
public interface IRequestHandler {
	
	/**
	 * 处理request请求
	 * @param request
	 * @param response
	 * @param e
	 * @return
	 */
	void handleRequest(HttpRequest request, HttpResponse response, MessageEvent e);
	
	/**
	 * 初始化
	 * @param 
	 * @return
	 */
	void init(RequestConfig config);
}

