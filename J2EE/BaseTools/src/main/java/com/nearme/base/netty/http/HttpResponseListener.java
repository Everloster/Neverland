/**
 * HttpResponseListener.java
 * com.nearme.base.netty.http
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-3-28 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.http;

import java.util.EventListener;

import org.jboss.netty.handler.codec.http.HttpResponse;

/**
 * ClassName:HttpResponseListener <br>
 * Function: http处理监听 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-3-28  上午9:34:02
 */
public interface HttpResponseListener extends EventListener {
	/**
	 * http返回成功处理
	 * @param
	 * @return
	 */
	public void onSuccess(HttpResponse response);

	/**
	 * http请求失败处理
	 * @param
	 * @return
	 */
	public void onError(HttpResponse response);
}

