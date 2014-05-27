/**
 * IHttpFactory.java
 * com.nearme.base.netty.server
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-6-3 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.gamecenter.ddz.oauth.server;

import java.io.InputStream;

/**
 * ClassName:IHttpFactory <br>
 * Function: http请求处理工厂，提供注册与获取处理器的功能N <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-6-3  下午3:44:45
 */
public interface IHttpFactory {

	/**
	 * 根据传入的url获取处理的实例
	 * @param requestUri url
	 * @return
	 */
	IRequestHandler getRequestHandler(String requestUri);

	/**
	 * 使用xml格式的配置文件注册
	 * @param
	 * @return
	 */
	boolean registerXml(InputStream inputStream);
}

